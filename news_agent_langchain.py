from __future__ import annotations

# Standard library imports first
import json
import os
import ssl
from typing import Any, Dict, List, Optional, Sequence

from dist_utils import should_run_on_this_rank

# Third-party imports
import requests
import urllib3
from urllib3.util.ssl_ import create_urllib3_context

# Forcing a more compatible SSL context to avoid UNEXPECTED_EOF_WHILE_READING
# This is a common workaround for this error on some systems/Python versions.
class LegacySSLAdapter(requests.adapters.HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        context = create_urllib3_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_REQUIRED
        # Use older TLS versions if necessary, or just a fresh context
        kwargs["ssl_context"] = context
        return super().init_poolmanager(*args, **kwargs)

# Try to patch the global session if huggingface_hub uses it, 
# or we can try to configure huggingface_hub's session.
from huggingface_hub import configure_http_backend

def _get_patched_session() -> requests.Session:
    session = requests.Session()
    adapter = LegacySSLAdapter()
    session.mount("https://", adapter)
    return session

configure_http_backend(backend_factory=_get_patched_session)

from huggingface_hub import InferenceClient
from huggingface_hub.errors import BadRequestError, HfHubHTTPError
from pydantic import BaseModel, Field, PrivateAttr

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool


HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_API_KEY")
HF_PROVIDER = os.getenv("HF_PROVIDER", "novita")
HF_MODEL = os.getenv("HF_MODEL", "zai-org/GLM-4.6")
HF_MAX_TOKENS = int(os.getenv("HF_MAX_TOKENS", "10000"))
HF_TEMPERATURE = float(os.getenv("HF_TEMPERATURE", "0.2"))
NEWS_API_KEY = os.getenv("NEWS_API_KEY")


class FetchTopHeadlinesArgs(BaseModel):
    country: str = Field(..., description="2-letter country code, e.g. 'us'.")
    category: str = Field(..., description="News category, e.g. 'technology'.")
    limit: int = Field(..., ge=1, le=10, description="Number of headlines.")


@tool("fetch_top_headlines", args_schema=FetchTopHeadlinesArgs)
def fetch_top_headlines(country: str, category: str, limit: int) -> str:
    """Fetch top headlines (live) from NewsAPI."""
    if not NEWS_API_KEY:
        return json.dumps({"error": "Missing NEWS_API_KEY"})

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
        return json.dumps({"error": f"NewsAPI request failed: {e}"})

    articles = data.get("articles") or []
    trimmed: List[Dict[str, Any]] = []
    for a in articles[:limit]:
        trimmed.append(
            {
                "title": a.get("title"),
                "description": a.get("description"),
                "url": a.get("url"),
                "source": (a.get("source") or {}).get("name"),
            }
        )

    return json.dumps({"articles": trimmed}, ensure_ascii=False)


def _parse_json_maybe(s: Any) -> Dict[str, Any]:
    if isinstance(s, str):
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            return {}
    if isinstance(s, dict):
        return s
    return {}


def _lc_messages_to_hf(messages: Sequence[BaseMessage]) -> List[Dict[str, Any]]:
    hf_messages: List[Dict[str, Any]] = []

    for m in messages:
        if isinstance(m, SystemMessage):
            hf_messages.append({"role": "system", "content": m.content})
            continue

        if isinstance(m, HumanMessage):
            hf_messages.append({"role": "user", "content": m.content})
            continue

        if isinstance(m, ToolMessage):
            hf_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": m.tool_call_id,
                    "content": m.content,
                }
            )
            continue

        if isinstance(m, AIMessage):
            d: Dict[str, Any] = {"role": "assistant", "content": m.content or ""}

            # Preserve tool calls in history (OpenAI-style format), so the provider can
            # match subsequent tool results by tool_call_id.
            tool_calls = getattr(m, "tool_calls", None) or []
            if tool_calls:
                openai_tool_calls: List[Dict[str, Any]] = []
                for i, tc in enumerate(tool_calls):
                    if isinstance(tc, dict):
                        tc_id = tc.get("id") or str(i)
                        tc_name = tc.get("name")
                        tc_args = tc.get("args")
                    else:
                        tc_id = getattr(tc, "id", None) or str(i)
                        tc_name = getattr(tc, "name", None)
                        tc_args = getattr(tc, "args", None)

                    if isinstance(tc_args, str):
                        arguments = tc_args
                    else:
                        arguments = json.dumps(tc_args or {}, ensure_ascii=False)

                    openai_tool_calls.append(
                        {
                            "id": tc_id,
                            "type": "function",
                            "function": {
                                "name": tc_name,
                                "arguments": arguments,
                            },
                        }
                    )
                d["tool_calls"] = openai_tool_calls

            hf_messages.append(d)
            continue

        # Fallback: treat unknown message type as a user message.
        hf_messages.append({"role": "user", "content": str(getattr(m, "content", m))})

    return hf_messages


class HFInferenceChatModel(BaseChatModel):
    """Minimal LangChain ChatModel wrapper over huggingface_hub.InferenceClient.chat_completion.

    This enables LangChain's tool-calling agents to work with Hugging Face Inference Providers
    (when the provider/model supports OpenAI-style tool calling).
    """

    model: str
    provider: str
    api_key: str
    max_tokens: int = 10000
    temperature: float = 0.2

    _client: InferenceClient = PrivateAttr()

    def __init__(self, **data: Any):
        super().__init__(**data)
        self._client = InferenceClient(provider=self.provider, api_key=self.api_key)

    def bind_tools(
        self,
        tools: Sequence[Any],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Bind tools for LangChain agents.

        LangChain v1.x agents call `model.bind_tools(...)` to attach tool schemas.
        Our HF chat wrapper accepts tool config via runtime kwargs, so we implement
        this by returning a bound runnable with those kwargs.
        """

        return self.bind(tools=tools, tool_choice=tool_choice or "auto", **kwargs)

    @property
    def _llm_type(self) -> str:
        return "huggingface_hub_inference"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"provider": self.provider, "model": self.model}

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        hf_messages = _lc_messages_to_hf(messages)

        raw_tools = kwargs.get("tools")
        tool_choice = kwargs.get("tool_choice", "auto")

        openai_tools = None
        if raw_tools:
            openai_tools = [convert_to_openai_tool(t) for t in raw_tools]

        params: Dict[str, Any] = {
            "model": self.model,
            "messages": hf_messages,
            "max_tokens": int(kwargs.get("max_tokens", self.max_tokens)),
            "temperature": float(kwargs.get("temperature", self.temperature)),
        }

        # Only send tool params when tools are actually bound.
        if openai_tools is not None:
            params["tools"] = openai_tools
            params["tool_choice"] = tool_choice

        # (Optional) stop sequences.
        if stop:
            params["stop"] = stop

        resp = self._client.chat_completion(**params)
        msg = resp.choices[0].message

        tool_calls: List[Dict[str, Any]] = []
        for tc in msg.tool_calls or []:
            tc = dict(tc)
            fn = tc.get("function") or {}
            name = fn.get("name")
            if not name:
                continue

            args = _parse_json_maybe(fn.get("arguments"))
            tool_calls.append(
                {
                    "type": "tool_call",
                    "id": tc.get("id"),
                    "name": name,
                    "args": args,
                }
            )

        ai = AIMessage(content=msg.content or "", tool_calls=tool_calls)
        return ChatResult(generations=[ChatGeneration(message=ai)])


def run_news_agent(country: str = "us", category: str = "technology", limit: int = 5) -> str:
    # In distributed/multi-process launches (torchrun/Slurm/MPI), default to rank-0 only to
    # avoid duplicating API calls and rate-limiting.
    if not should_run_on_this_rank():
        return ""

    if not HF_TOKEN:
        return "Missing HF_TOKEN (or HF_API_KEY). You need a HF token with Inference Providers permission."

    llm = HFInferenceChatModel(
        provider=HF_PROVIDER,
        api_key=HF_TOKEN,
        model=HF_MODEL,
        max_tokens=HF_MAX_TOKENS,
        temperature=HF_TEMPERATURE,
    )

    tools = [fetch_top_headlines]

    graph = create_agent(
        model=llm,
        tools=tools,
        system_prompt=(
            "You are a news assistant. You MUST use the provided tool to fetch real headlines. "
            "Do not invent headlines. After you receive tool results, summarize them clearly."
        ),
    )

    user_input = (
        f"Get the top {limit} {category} headlines for country={country}.\n"
        "Return Markdown with one section per headline:\n"
        "- Title\n"
        "- 2-sentence summary (use description + title)\n"
        "- One bullet: 'Why it matters'\n"
        "Include the URL if available."
    )

    try:
        state = graph.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config={"recursion_limit": 12},
        )

        messages = state.get("messages", []) if isinstance(state, dict) else []
        for m in reversed(messages):
            if isinstance(m, AIMessage):
                return m.content or ""
        return ""
    except (BadRequestError, HfHubHTTPError):
        # Fallback: call the tool in Python and ask the model to format + summarize.
        tool_output = fetch_top_headlines.invoke({"country": country, "category": category, "limit": limit})
        tool_json = _parse_json_maybe(tool_output)
        if "error" in tool_json:
            return str(tool_json["error"])

        safe_json = json.dumps(tool_json, ensure_ascii=False)
        messages: List[BaseMessage] = [
            SystemMessage(
                content=(
                    "You are a news assistant. Use ONLY the provided JSON (no outside knowledge). "
                    "Do not invent headlines or URLs."
                )
            ),
            HumanMessage(
                content=(
                    "Here are the live headlines as JSON:\n\n"
                    f"{safe_json}\n\n"
                    "Return Markdown with one section per headline:\n"
                    "- Title\n"
                    "- 2-sentence summary (use description + title)\n"
                    "- One bullet: 'Why it matters'\n"
                    "Include the URL if available."
                )
            ),
        ]
        return llm.invoke(messages).content or ""


if __name__ == "__main__":
    if should_run_on_this_rank():
        print("Daily Tech Headlines:\n")
        print(run_news_agent(country="us", category="technology", limit=5))
