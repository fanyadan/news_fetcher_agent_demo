from __future__ import annotations

# Standard library imports first
import json
import os
import ssl
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dist_utils import (
    get_distributed_context,
    is_truthy_env,
    mpi_sanity_check,
    should_run_on_this_rank,
)

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
from huggingface_hub import utils

configure_http_backend = getattr(utils, "configure_http_backend", None)
if configure_http_backend is None:
    # Try older location or direct import if it exists but not in __init__
    try:
        from huggingface_hub.utils import configure_http_backend as _configure_http_backend
    except ImportError:
        _configure_http_backend = None

    configure_http_backend = _configure_http_backend


def _get_patched_session() -> requests.Session:
    session = requests.Session()
    adapter = LegacySSLAdapter()
    session.mount("https://", adapter)
    return session


# `configure_http_backend` is not available in all huggingface_hub versions.
# Guard the call to avoid failing at import time.
if callable(configure_http_backend):
    configure_http_backend(backend_factory=_get_patched_session)

from huggingface_hub import InferenceClient
from huggingface_hub.errors import BadRequestError, HfHubHTTPError
try:
    from pydantic import BaseModel, Field, PrivateAttr
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pydantic"])
    from pydantic import BaseModel, Field, PrivateAttr

try:
    from langchain.agents import create_agent
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_core.tools import tool
    from langchain_core.utils.function_calling import convert_to_openai_tool
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "langchain", "langchain-core"])
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
    page: int = Field(1, ge=1, description="Page number (1-indexed).")


@tool("fetch_top_headlines", args_schema=FetchTopHeadlinesArgs)
def fetch_top_headlines(country: str, category: str, limit: int, page: int = 1) -> str:
    """Fetch top headlines (live) from NewsAPI."""
    if not NEWS_API_KEY:
        return json.dumps({"error": "Missing NEWS_API_KEY"})

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


def _summarize_one_article_llm(llm: HFInferenceChatModel, *, article: Dict[str, Any]) -> Dict[str, str]:
    title = (article.get("title") or "").strip()
    description = (article.get("description") or "").strip()
    source = (article.get("source") or "").strip()

    messages: List[BaseMessage] = [
        SystemMessage(
            content=(
                "You are a news assistant. Use ONLY the provided title/description (no outside knowledge). "
                "Return a JSON object with keys: summary (exactly 2 sentences), why_it_matters (1 sentence). "
                "Return JSON only."
            )
        ),
        HumanMessage(content=f"Title: {title}\nDescription: {description}\nSource: {source}\n"),
    ]

    content = (llm.invoke(messages).content or "").strip()
    obj = _extract_json_object(content)
    summary = (obj.get("summary") or "").strip()
    why = (obj.get("why_it_matters") or obj.get("why") or "").strip()

    if not summary:
        summary = content

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


def _run_news_agent_torch_sharded(country: str, category: str, limit: int, *, llm: HFInferenceChatModel) -> str:
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
        payload_str = fetch_top_headlines.invoke({"country": country, "category": category, "limit": requested_limit, "page": 1})
        payload = _parse_json_maybe(payload_str)
        if not isinstance(payload, dict):
            payload = {"error": f"Unexpected payload type from fetch_top_headlines: {type(payload)}"}
        if not isinstance(payload.get("articles"), list):
            payload.setdefault("articles", [])

    obj_list: List[Any] = [payload]
    dist.broadcast_object_list(obj_list, src=0)
    payload_b = obj_list[0] if isinstance(obj_list[0], dict) else {}

    err_msg = payload_b.get("error") if isinstance(payload_b, dict) else None
    articles_any = payload_b.get("articles") if isinstance(payload_b, dict) else []
    articles: List[Any] = articles_any if isinstance(articles_any, list) else []

    local_sections: List[Dict[str, Any]] = []
    if not err_msg:
        for idx, art_any in enumerate(articles):
            if idx % world_size != rank:
                continue

            art = art_any if isinstance(art_any, dict) else {}
            try:
                out = _summarize_one_article_llm(llm, article=art)
                section_md = _format_article_section(article=art, summary=out["summary"], why_it_matters=out["why_it_matters"])
            except Exception as e:
                title = (art.get("title") or "(untitled)").strip()
                section_md = f"## {title}\n\nFailed to summarize on rank {rank}: {e}"

            local_sections.append({"index": idx, "markdown": section_md})

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


def _run_news_agent_mpi_sharded(country: str, category: str, limit: int, *, llm: HFInferenceChatModel) -> str:
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

    if is_truthy_env("NEWS_AGENT_MPI_CHECK"):
        chk_err = mpi_sanity_check(comm, verbose=True, tag="news_agent_langchain")
        if chk_err:
            return chk_err if rank == 0 else ""

    requested_limit = int(limit)

    payload: Dict[str, Any] = {}
    if rank == 0:
        payload_str = fetch_top_headlines.invoke({"country": country, "category": category, "limit": requested_limit, "page": 1})
        payload = _parse_json_maybe(payload_str)
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
        for idx, art_any in enumerate(articles):
            if idx % world_size != rank:
                continue

            art = art_any if isinstance(art_any, dict) else {}
            try:
                out = _summarize_one_article_llm(llm, article=art)
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

        llm_shard = HFInferenceChatModel(
            provider=HF_PROVIDER,
            api_key=HF_TOKEN,
            model=HF_MODEL,
            max_tokens=min(HF_MAX_TOKENS, 600),
            temperature=HF_TEMPERATURE,
        )

        backend = _distributed_backend()

        if backend in {"mpi", "mpi4py"}:
            return _run_news_agent_mpi_sharded(country=country, category=category, limit=limit, llm=llm_shard)

        if backend == "auto":
            if _looks_like_mpi_launch():
                return _run_news_agent_mpi_sharded(country=country, category=category, limit=limit, llm=llm_shard)
            return _run_news_agent_torch_sharded(country=country, category=category, limit=limit, llm=llm_shard)

        # default: torch
        return _run_news_agent_torch_sharded(country=country, category=category, limit=limit, llm=llm_shard)

    # Default behavior: rank-0 only (unless mode=all).
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
    md = run_news_agent(country="us", category="technology", limit=10)
    if should_run_on_this_rank():
        print("Daily Tech Headlines:\n")
        print(md)
