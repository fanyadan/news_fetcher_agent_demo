# News Agent (Hugging Face + NewsAPI)

This repo contains two small “news agent” demos:

- `news_agent_hf_toolcall.py`: uses `huggingface_hub.InferenceClient` directly (no LangChain).
- `news_agent_langchain.py`: uses a LangChain agent graph (`langchain.agents.create_agent`) with a small Hugging Face chat-model wrapper.

Both scripts:

- Fetch **live** headlines from **NewsAPI** (`/v2/top-headlines`)
- Use a Hugging Face-hosted LLM (via `huggingface_hub.InferenceClient`) to **summarize and format** the headlines
- Attempt **OpenAI-style tool/function calling** first, and fall back to a manual JSON-to-summary flow if the model/provider doesn’t support tool calls

The output is a Markdown digest with one section per headline:

- Title
- 2-sentence summary (based on title + description)
- One bullet: “Why it matters”
- URL (if available)

## Requirements

Common (both scripts):

- Python 3.9+
- A **NewsAPI** key
- A **Hugging Face** token with access to your chosen Inference Provider/model
- Python packages:
  - `requests`
  - `huggingface_hub`

Optional (only for `NEWS_AGENT_DISTRIBUTED_MODE=shard`):

- `NEWS_AGENT_DISTRIBUTED_BACKEND=torch`:
  - PyTorch (`torch`) with distributed support
  - `numpy` (some torch distributed object collectives can require NumPy)
- `NEWS_AGENT_DISTRIBUTED_BACKEND=mpi`:
  - `mpi4py`

For `news_agent_langchain.py` (LangChain agent):

- Python packages:
  - `langchain`
  - `langgraph`
  - `pydantic`

Install dependencies:

```bash
pip install -U requests huggingface_hub

# Only needed for the LangChain version
pip install -U langchain langgraph pydantic

# Only needed for shard mode (torch backend)
pip install -U numpy

# Only needed for shard mode (MPI backend)
pip install -U mpi4py
```

## Configuration

The script reads the following environment variables:

### NewsAPI

- `NEWS_API_KEY` (required)

### Hugging Face

- `HF_TOKEN` (required) — also accepts `HUGGINGFACEHUB_API_TOKEN` or `HF_API_KEY`
- `HF_PROVIDER` (optional, default: `novita`)
- `HF_MODEL` (optional, default: `zai-org/GLM-4.6`)

Only for `news_agent_langchain.py`:

- `HF_MAX_TOKENS` (optional, default: `10000`)
- `HF_TEMPERATURE` (optional, default: `0.2`)

### Distributed / multi-process execution

These scripts do **not** run a local PyTorch model. By default, they call a hosted LLM via `huggingface_hub.InferenceClient` (remote inference over HTTP).

In `NEWS_AGENT_DISTRIBUTED_MODE=shard`, ranks coordinate **optionally** using either **`torch.distributed`** (default, `NEWS_AGENT_DISTRIBUTED_BACKEND=torch`) or **MPI via `mpi4py`** (`NEWS_AGENT_DISTRIBUTED_BACKEND=mpi`). This does *not* turn the Hugging Face model into “distributed PyTorch inference”.

If you run these scripts under a multi-process launcher (e.g. `torchrun`, Slurm, MPI), they default to **rank 0 only** to avoid duplicating external API calls.

- `NEWS_AGENT_DISTRIBUTED_MODE` (optional, default: `rank0`)
  - `rank0`: only rank 0 runs
  - `all`: every rank runs the full pipeline (duplicates work)
  - `shard`: sharded summarization mode.
    - Enabled only when **(1)** `NEWS_AGENT_DISTRIBUTED_MODE=shard` (or `torch` / `torch_shard`) **and (2)** `WORLD_SIZE > 1` (i.e., you actually launched multiple ranks).
    - Rank 0 fetches the headlines **once**, broadcasts the article list to all ranks, and each rank summarizes a slice (round-robin by index).
    - Rank 0 gathers per-rank results and assembles the final Markdown digest.
    - Effective headlines fetched = up to `limit` (whatever NewsAPI returns).
    - Total HF inference calls are ~`limit` (spread across ranks). Provider rate limits still apply.
- `NEWS_AGENT_DISTRIBUTED_BACKEND` (optional, default: `torch`)
  - `torch`: use `torch.distributed` collectives
  - `mpi`: use `mpi4py` collectives
  - `auto`: prefer `mpi` when launched under MPI env vars; otherwise use `torch`
- `NEWS_AGENT_TORCH_BACKEND` (optional, default: `gloo`) — only used when `NEWS_AGENT_DISTRIBUTED_BACKEND=torch`
- Manual overrides (optional): `NEWS_AGENT_RANK`, `NEWS_AGENT_WORLD_SIZE`, `NEWS_AGENT_LOCAL_RANK`.

Example (sharded mode with torchrun):

```bash
export NEWS_AGENT_DISTRIBUTED_MODE=shard
export NEWS_AGENT_DISTRIBUTED_BACKEND=torch

# Launch multiple ranks; headlines are fetched once and summarization is sharded across ranks.
torchrun --standalone --nproc_per_node=4 news_agent_hf_toolcall.py
```

You can run the LangChain version the same way:

```bash
torchrun --standalone --nproc_per_node=4 news_agent_langchain.py
```

Example (sharded mode with MPI):

```bash
export NEWS_AGENT_DISTRIBUTED_MODE=shard
export NEWS_AGENT_DISTRIBUTED_BACKEND=mpi

mpirun -n 4 python news_agent_hf_toolcall.py
```

> Note: Many large models are not available on the free `hf-inference` (serverless) tier and can return 404. Using an Inference Provider (like `novita`) and a compatible model is often required.

Example:

```bash
export NEWS_API_KEY="your_newsapi_key"
export HF_TOKEN="your_hf_token"

# Optional overrides
export HF_PROVIDER="novita"
export HF_MODEL="zai-org/GLM-4.6"
```

## Run

No LangChain (direct Hugging Face calls):

```bash
python news_agent_hf_toolcall.py
```

LangChain agent version:

```bash
python news_agent_langchain.py
```

By default, the script prints “Daily Tech Headlines” for:

- `country="us"`
- `category="technology"`
- `limit=5`

## Programmatic usage

### `news_agent_hf_toolcall.py`

```python
from news_agent_hf_toolcall import run_news_agent

md = run_news_agent(country="us", category="technology", limit=5)
print(md)
```

### `news_agent_langchain.py`

```python
from news_agent_langchain import run_news_agent

md = run_news_agent(country="us", category="technology", limit=5)
print(md)
```

Parameters:

- `country`: 2-letter country code (e.g. `"us"`)
- `category`: NewsAPI category (e.g. `"technology"`, `"business"`, `"sports"`)
- `limit`: number of headlines (int). The tool schema caps tool-called requests at **10**.
  - In `NEWS_AGENT_DISTRIBUTED_MODE=shard`, rank 0 fetches headlines once and ranks shard summarization; increasing `WORLD_SIZE` increases parallelism.

## How it works

### Tool: `fetch_top_headlines`

- Implemented in `fetch_top_headlines(country, category, limit, page=1)`
- Calls `https://newsapi.org/v2/top-headlines` (supports pagination via the `page` parameter)
- Returns a trimmed JSON payload containing:
  - `title`
  - `description`
  - `url`
  - `source`

### `news_agent_hf_toolcall.py` (no LangChain)

`run_news_agent()`:

1. Sends a system + user prompt, plus `TOOLS` (OpenAI-style tool schema), to `InferenceClient.chat_completion(...)`.
2. If the model returns `tool_calls`, Python executes them and appends `role="tool"` messages containing the JSON output.
3. The model then generates the final Markdown summary.

### `news_agent_langchain.py` (LangChain agent)

- Wraps `huggingface_hub.InferenceClient.chat_completion(...)` in a minimal LangChain `BaseChatModel` (`HFInferenceChatModel`).
- Builds an agent graph with `langchain.agents.create_agent(...)`.
- LangChain drives the tool-calling loop by reading `AIMessage.tool_calls`, executing tools, and feeding tool results back as `ToolMessage`.

### Fallback mode

If tool calling isn’t supported (or certain HTTP errors occur), both scripts fall back to:

1. Call `fetch_top_headlines(...)` directly in Python.
2. Send the resulting JSON to the model and instruct it to use **only** that JSON (no outside knowledge).

## Troubleshooting

### `Missing NEWS_API_KEY`

Set `NEWS_API_KEY`:

```bash
export NEWS_API_KEY="..."
```

### `Missing HF_TOKEN (or HF_API_KEY)`

Set `HF_TOKEN` (or `HUGGINGFACEHUB_API_TOKEN` / `HF_API_KEY`):

```bash
export HF_TOKEN="..."
```

### `RuntimeError: Numpy is not available`

If you run with `NEWS_AGENT_DISTRIBUTED_MODE=shard` and `NEWS_AGENT_DISTRIBUTED_BACKEND=torch`, PyTorch distributed object collectives can require NumPy.

Fix:

```bash
pip install -U numpy
```

### Missing LangChain dependencies

If you see import errors for `langchain` / `langgraph`, install the LangChain deps:

```bash
pip install -U langchain langgraph pydantic
```

### 404 / model not found / provider mismatch

Try:

- Setting `HF_PROVIDER` to a provider you have enabled
- Switching `HF_MODEL` to a model supported by that provider

### NewsAPI errors / rate limits

Common causes:

- Invalid API key
- Rate limiting
- Unsupported `country`/`category`
- Network issues

## Security notes

- Treat `NEWS_API_KEY` and `HF_TOKEN` as secrets.
- Prefer environment variables (as shown above) over hard-coding tokens in source files.
