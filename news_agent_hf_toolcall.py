import json
import os
from typing import Any, Dict, List

import requests
from huggingface_hub import InferenceClient
from huggingface_hub.errors import BadRequestError, HfHubHTTPError

from dist_utils import rank0_print, should_run_on_this_rank


HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_API_KEY")
# NOTE: Many large LLMs are not hosted on the free `hf-inference` (serverless) tier and will 404.
# If you have Inference Providers enabled on your HF token, use a provider like `novita`.
HF_PROVIDER = os.getenv("HF_PROVIDER", "novita")
#HF_MODEL = os.getenv("HF_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
HF_MODEL = os.getenv("HF_MODEL", "zai-org/GLM-4.6")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")


def fetch_top_headlines(country: str, category: str, limit: int) -> Dict[str, Any]:
    """
    Tool: fetch live headlines from NewsAPI.
    """
    if not NEWS_API_KEY:
        return {"error": "Missing NEWS_API_KEY"}

    url = "https://newsapi.org/v2/top-headlines"
    params = {
        "apiKey": NEWS_API_KEY,
        "country": country,
        "category": category,
        "pageSize": limit,
    }

    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return {"error": f"NewsAPI request failed: {e}"}

    articles = data.get("articles") or []
    trimmed = []
    for a in articles[:limit]:
        trimmed.append(
            {
                "title": a.get("title"),
                "description": a.get("description"),
                "url": a.get("url"),
                "source": (a.get("source") or {}).get("name"),
            }
        )
    return {"articles": trimmed}


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "fetch_top_headlines",
            "description": "Fetch top headlines (live) from NewsAPI.",
            "parameters": {
                "type": "object",
                "properties": {
                    "country": {"type": "string", "description": "2-letter country code, e.g. 'us'."},
                    "category": {"type": "string", "description": "News category, e.g. 'technology'."},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 10, "description": "Number of headlines."},
                },
                "required": ["country", "category", "limit"],
                "additionalProperties": False,
            },
        },
    }
]

TOOL_IMPL = {
    "fetch_top_headlines": fetch_top_headlines,
}


def _parse_tool_args(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    fn = tool_call.get("function") or {}
    args = fn.get("arguments", {})
    if isinstance(args, str):
        try:
            return json.loads(args)
        except json.JSONDecodeError:
            return {}
    if args is None:
        return {}
    return args


def run_news_agent(country: str = "us", category: str = "technology", limit: int = 5) -> str:
    # In distributed/multi-process launches (torchrun/Slurm/MPI), default to rank-0 only to
    # avoid duplicating API calls and rate-limiting.
    if not should_run_on_this_rank():
        return ""

    if not HF_TOKEN:
        return "Missing HF_TOKEN (or HF_API_KEY). You need a HF token with Inference Providers permission."

    client = InferenceClient(provider=HF_PROVIDER, api_key=HF_TOKEN)

    # First try OpenAI-style function calling (tools/tool_choice). Not all providers/models support it.
    messages: List[Dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are a news assistant. You MUST use the provided tool to fetch real headlines. "
                "Do not invent headlines. After you receive tool results, summarize them clearly."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Get the top {limit} {category} headlines for country={country}.\n"
                "Return Markdown with one section per headline:\n"
                "- Title\n"
                "- 2-sentence summary (use description + title)\n"
                "- One bullet: 'Why it matters'\n"
                "Include the URL if available."
            ),
        },
    ]

    fallback_to_manual = False

    for _ in range(6):  # max tool-call turns
        try:
            resp = client.chat_completion(
                model=HF_MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",  # set to "required" if you want to force at least one tool call
                max_tokens=10000,
                temperature=0.2,
            )
        except BadRequestError as e:
            # Some providers/models don't support function calling.
            details = (getattr(e, "response", None).text or "") if getattr(e, "response", None) is not None else ""
            if "function" in details.lower() and "call" in details.lower():
                fallback_to_manual = True
                break
            return f"HF chat_completion failed: {e}"
        except HfHubHTTPError as e:
            # hf-inference (serverless) returns 404 for models that are not hosted/warm.
            if getattr(e, "response", None) is not None and e.response.status_code == 404:
                fallback_to_manual = True
                break
            return f"HF chat_completion failed: {e}"
        except Exception as e:
            return f"HF chat_completion failed: {e}"

        msg = resp.choices[0].message
        tool_calls = msg.tool_calls or []

        if tool_calls:
            # 1) Add the assistant tool-call message to history
            messages.append(
                {
                    "role": msg.role,
                    "content": msg.content,
                    "tool_calls": [dict(tc) for tc in tool_calls],
                }
            )

            # 2) Execute each tool call and add tool outputs
            for tc in tool_calls:
                tc = dict(tc)
                tool_name = (tc.get("function") or {}).get("name")
                args = _parse_tool_args(tc)

                fn = TOOL_IMPL.get(tool_name)
                if fn is None:
                    output = {"error": f"Unknown tool: {tool_name}"}
                else:
                    output = fn(**args)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.get("id"),
                        "content": json.dumps(output),
                    }
                )

            # 3) Continue loop (model now sees tool outputs)
            continue

        # No tool calls => final answer
        return msg.content or ""

    # Fallback: call the tool in Python and ask the model to format + summarize.
    if fallback_to_manual:
        rank0_print("The model does not support function-call, fallback to manual\n")
        tool_output = fetch_top_headlines(country=country, category=category, limit=limit)
        if "error" in tool_output:
            return str(tool_output["error"])

        safe_json = json.dumps(tool_output, ensure_ascii=False)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a news assistant. Use ONLY the provided JSON (no outside knowledge). "
                    "Do not invent headlines or URLs."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Here are the live headlines as JSON:\n\n"
                    f"{safe_json}\n\n"
                    "Return Markdown with one section per headline:\n"
                    "- Title\n"
                    "- 2-sentence summary (use description + title)\n"
                    "- One bullet: 'Why it matters'\n"
                    "Include the URL if available."
                ),
            },
        ]

        try:
            resp = client.chat_completion(
                model=HF_MODEL,
                messages=messages,
                max_tokens=10000,
                temperature=0.2,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            return f"HF chat_completion failed (fallback): {e}"

    return "Max steps reached without a final answer."


if __name__ == "__main__":
    if should_run_on_this_rank():
        print("Daily Tech Headlines:\n")
        print(run_news_agent(country="us", category="technology", limit=5))
