import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

import requests
try:
    from huggingface_hub import InferenceClient
    from huggingface_hub.errors import BadRequestError, HfHubHTTPError
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
    from huggingface_hub import InferenceClient
    from huggingface_hub.errors import BadRequestError, HfHubHTTPError

from dist_utils import get_distributed_context, rank0_print, should_run_on_this_rank


HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_API_KEY")
# NOTE: Many large LLMs are not hosted on the free `hf-inference` (serverless) tier and will 404.
# If you have Inference Providers enabled on your HF token, use a provider like `novita`.
HF_PROVIDER = os.getenv("HF_PROVIDER", "novita")
#HF_MODEL = os.getenv("HF_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
HF_MODEL = os.getenv("HF_MODEL", "zai-org/GLM-4.6")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")


def fetch_top_headlines(country: str, category: str, limit: int, page: int = 1) -> Dict[str, Any]:
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
        "page": page,
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
                    "page": {"type": "integer", "minimum": 1, "description": "Page number (1-indexed)."},
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


def _distributed_mode() -> str:
    return (os.getenv("NEWS_AGENT_DISTRIBUTED_MODE") or "rank0").strip().lower()


def _distributed_backend() -> str:
    """Select the sharding communication backend.

    - torch: use torch.distributed collectives
    - mpi: use mpi4py collectives
    """

    return (os.getenv("NEWS_AGENT_DISTRIBUTED_BACKEND") or "torch").strip().lower()


def _looks_like_mpi_launch() -> bool:
    # If torchrun env vars are set, prefer torch.distributed in backend=auto.
    for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
        if os.getenv(k):
            return False

    # Common MPI environment variables (OpenMPI/PMI/MVAPICH).
    for k in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "MV2_COMM_WORLD_SIZE"):
        if os.getenv(k):
            return True
    return False


def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if not s.startswith("```"):
        return s
    lines = s.splitlines()
    if len(lines) >= 2 and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _extract_json_object(text: str) -> Dict[str, Any]:
    text = _strip_code_fences(text)
    text = text.strip()
    if not text:
        return {}

    # Best-effort JSON extraction (handles extra pre/post text).
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}

    candidate = text[start : end + 1]
    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        return {}


def _try_init_torch_distributed(world_size: int) -> Tuple[Optional[object], Optional[str]]:
    if world_size <= 1:
        return None, None

    try:
        import torch.distributed as dist  # type: ignore
    except Exception as e:
        return None, f"Shard mode requires torch.distributed, but import failed: {e}"

    if not getattr(dist, "is_available", lambda: False)():
        return None, "Shard mode requires torch.distributed, but it is not available in this torch build."

    if not dist.is_initialized():
        backend = os.getenv("NEWS_AGENT_TORCH_BACKEND", "gloo")
        try:
            dist.init_process_group(backend=backend, init_method="env://")
        except Exception as e:
            return None, f"torch.distributed init_process_group failed: {e}"

    return dist, None


def _summarize_one_article(client: InferenceClient, *, article: Dict[str, Any]) -> Dict[str, str]:
    title = (article.get("title") or "").strip()
    description = (article.get("description") or "").strip()
    source = (article.get("source") or "").strip()

    messages = [
        {
            "role": "system",
            "content": (
                "You are a news assistant. Use ONLY the provided title/description (no outside knowledge). "
                "Return a JSON object with keys: summary (exactly 2 sentences), why_it_matters (1 sentence). "
                "Return JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Title: {title}\n"
                f"Description: {description}\n"
                f"Source: {source}\n"
            ),
        },
    ]

    resp = client.chat_completion(
        model=HF_MODEL,
        messages=messages,
        max_tokens=300,
        temperature=0.2,
    )

    text = (resp.choices[0].message.content or "").strip()
    obj = _extract_json_object(text)
    summary = (obj.get("summary") or "").strip()
    why = (obj.get("why_it_matters") or obj.get("why") or "").strip()

    if not summary:
        # Fallback: take the raw text as the summary.
        summary = text

    return {"summary": summary, "why_it_matters": why}


def _format_article_section(*, article: Dict[str, Any], summary: str, why_it_matters: str) -> str:
    title = (article.get("title") or "(untitled)").strip()
    url = (article.get("url") or "").strip()

    lines: List[str] = [f"## {title}", summary.strip()]

    if why_it_matters.strip():
        lines.append(f"- Why it matters: {why_it_matters.strip()}")
    else:
        lines.append("- Why it matters: (not provided)")

    if url:
        lines.append(url)

    return "\n\n".join([x for x in lines if x]).strip()


def _run_news_agent_torch_sharded(country: str, category: str, limit: int) -> str:
    """Sharded execution via torch.distributed.

    Rank 0 fetches the headlines once, broadcasts the list of articles to all ranks,
    and then each rank summarizes a slice of the articles. Summaries are gathered and
    assembled on rank 0.
    """

    ctx = get_distributed_context()
    dist, err = _try_init_torch_distributed(ctx.world_size)
    if err:
        return err if ctx.is_main else ""

    if dist is None or not dist.is_initialized():
        return ""

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    requested_limit = int(limit)

    payload: Dict[str, Any] = {}
    if rank == 0:
        payload = fetch_top_headlines(country=country, category=category, limit=requested_limit, page=1)
        if not isinstance(payload, dict):
            payload = {"error": f"Unexpected payload type from fetch_top_headlines: {type(payload)}"}
        if not isinstance(payload.get("articles"), list):
            payload.setdefault("articles", [])

    # Broadcast the fetched articles (or error) so every rank stays in lock-step.
    obj_list: List[Any] = [payload]
    dist.broadcast_object_list(obj_list, src=0)
    payload_b = obj_list[0] if isinstance(obj_list[0], dict) else {}

    err_msg = payload_b.get("error") if isinstance(payload_b, dict) else None
    articles_any = payload_b.get("articles") if isinstance(payload_b, dict) else []
    articles: List[Any] = articles_any if isinstance(articles_any, list) else []

    local_sections: List[Dict[str, Any]] = []
    if not err_msg:
        client = InferenceClient(provider=HF_PROVIDER, api_key=HF_TOKEN)
        for idx, art_any in enumerate(articles):
            if idx % world_size != rank:
                continue

            art = art_any if isinstance(art_any, dict) else {}
            try:
                out = _summarize_one_article(client, article=art)
                section_md = _format_article_section(article=art, summary=out["summary"], why_it_matters=out["why_it_matters"])
            except Exception as e:
                title = (art.get("title") or "(untitled)").strip()
                section_md = f"## {title}\n\nFailed to summarize on rank {rank}: {e}"

            local_sections.append({"index": idx, "markdown": section_md})

    # Gather sections to rank 0.
    all_sections_by_rank: List[Any]
    if hasattr(dist, "gather_object"):
        gather_list = [None for _ in range(world_size)] if rank == 0 else None
        dist.gather_object(local_sections, gather_list, dst=0)
        all_sections_by_rank = gather_list or []
    else:
        gather_list = [None for _ in range(world_size)]
        dist.all_gather_object(gather_list, local_sections)
        all_sections_by_rank = gather_list

    if rank != 0:
        return ""

    if err_msg:
        return str(err_msg)

    merged: List[Dict[str, Any]] = []
    for chunk in all_sections_by_rank:
        if isinstance(chunk, list):
            merged.extend([x for x in chunk if isinstance(x, dict)])

    merged.sort(key=lambda x: int(x.get("index", 0)))

    header = f"# Top {len(articles)} {category.title()} Headlines ({country.upper()})"
    body = "\n\n".join([str(x.get("markdown") or "").strip() for x in merged if x.get("markdown")])

    parts = [header]
    if body:
        parts.append(body)

    return "\n\n".join(parts).strip()


def _try_init_mpi(world_size: int) -> Tuple[Optional[object], Optional[str]]:
    if world_size <= 1:
        return None, None

    try:
        from mpi4py import MPI  # type: ignore
    except Exception as e:
        return None, f"Shard mode requires mpi4py, but import failed: {e}"

    comm = getattr(MPI, "COMM_WORLD", None)
    if comm is None:
        return None, "mpi4py is installed, but MPI.COMM_WORLD is not available."

    return comm, None


def _run_news_agent_mpi_sharded(country: str, category: str, limit: int) -> str:
    """Sharded execution via MPI (mpi4py).

    Rank 0 fetches the headlines once, broadcasts the list of articles to all ranks,
    and then each rank summarizes a slice of the articles. Summaries are gathered and
    assembled on rank 0.
    """

    ctx = get_distributed_context()
    comm, err = _try_init_mpi(ctx.world_size)
    if err:
        return err if ctx.is_main else ""

    if comm is None:
        return ""

    try:
        rank = int(comm.Get_rank())
        world_size = int(comm.Get_size())
    except Exception as e:
        return f"MPI initialization failed: {e}" if ctx.is_main else ""

    if world_size <= 1:
        return (
            "MPI backend selected but MPI world_size=1. Launch with mpirun/srun (multiple ranks), "
            "or set NEWS_AGENT_DISTRIBUTED_BACKEND=torch."
            if ctx.is_main
            else ""
        )

    requested_limit = int(limit)

    payload: Dict[str, Any] = {}
    if rank == 0:
        payload = fetch_top_headlines(country=country, category=category, limit=requested_limit, page=1)
        if not isinstance(payload, dict):
            payload = {"error": f"Unexpected payload type from fetch_top_headlines: {type(payload)}"}
        if not isinstance(payload.get("articles"), list):
            payload.setdefault("articles", [])

    payload_b = comm.bcast(payload, root=0)
    payload_b = payload_b if isinstance(payload_b, dict) else {}

    err_msg = payload_b.get("error")
    articles_any = payload_b.get("articles")
    articles: List[Any] = articles_any if isinstance(articles_any, list) else []

    local_sections: List[Dict[str, Any]] = []
    if not err_msg:
        client = InferenceClient(provider=HF_PROVIDER, api_key=HF_TOKEN)
        for idx, art_any in enumerate(articles):
            if idx % world_size != rank:
                continue

            art = art_any if isinstance(art_any, dict) else {}
            try:
                out = _summarize_one_article(client, article=art)
                section_md = _format_article_section(article=art, summary=out["summary"], why_it_matters=out["why_it_matters"])
            except Exception as e:
                title = (art.get("title") or "(untitled)").strip()
                section_md = f"## {title}\n\nFailed to summarize on rank {rank}: {e}"

            local_sections.append({"index": idx, "markdown": section_md})

    all_sections_by_rank = comm.gather(local_sections, root=0)

    if rank != 0:
        return ""

    if err_msg:
        return str(err_msg)

    merged: List[Dict[str, Any]] = []
    if isinstance(all_sections_by_rank, list):
        for chunk in all_sections_by_rank:
            if isinstance(chunk, list):
                merged.extend([x for x in chunk if isinstance(x, dict)])

    merged.sort(key=lambda x: int(x.get("index", 0)))

    header = f"# Top {len(articles)} {category.title()} Headlines ({country.upper()})"
    body = "\n\n".join([str(x.get("markdown") or "").strip() for x in merged if x.get("markdown")])

    parts = [header]
    if body:
        parts.append(body)

    return "\n\n".join(parts).strip()


def run_news_agent(country: str = "us", category: str = "technology", limit: int = 5) -> str:
    mode = _distributed_mode()
    ctx = get_distributed_context()

    # Sharded distributed mode: rank0 fetches once, ranks shard summarization,
    # then results are gathered and assembled on rank0.
    if mode in {"shard", "torch", "torch_shard", "torch-shard"} and ctx.world_size > 1:
        if not HF_TOKEN:
            return "Missing HF_TOKEN (or HF_API_KEY). You need a HF token with Inference Providers permission." if ctx.is_main else ""

        backend = _distributed_backend()

        if backend in {"mpi", "mpi4py"}:
            return _run_news_agent_mpi_sharded(country=country, category=category, limit=limit)

        if backend == "auto":
            if _looks_like_mpi_launch():
                return _run_news_agent_mpi_sharded(country=country, category=category, limit=limit)
            return _run_news_agent_torch_sharded(country=country, category=category, limit=limit)

        # default: torch
        return _run_news_agent_torch_sharded(country=country, category=category, limit=limit)

    # Default behavior: rank-0 only (unless mode=all).
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
    md = run_news_agent(country="us", category="technology", limit=10)
    if should_run_on_this_rank():
        print("Daily Tech Headlines:\n")
        print(md)
