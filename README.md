# News Agent (Hugging Face tool calling + NewsAPI)

`news_agent_hf_toolcall.py` is a small “news agent” demo that:

- Fetches **live** headlines from **NewsAPI** (`/v2/top-headlines`)
- Uses a Hugging Face-hosted LLM (via `huggingface_hub.InferenceClient`) to **summarize and format** the headlines
- Attempts **OpenAI-style tool/function calling** first, and falls back to a manual JSON-to-summary flow if the model/provider doesn’t support tool calls

The output is a Markdown digest with one section per headline:

- Title
- 2-sentence summary (based on title + description)
- One bullet: “Why it matters”
- URL (if available)

## Requirements

- Python 3.9+
- A **NewsAPI** key
- A **Hugging Face** token with access to your chosen Inference Provider/model
- Python packages:
  - `requests`
  - `huggingface_hub`

Install dependencies:

```bash
pip install -U requests huggingface_hub
```

## Configuration

The script reads the following environment variables:

### NewsAPI

- `NEWS_API_KEY` (required)

### Hugging Face

- `HF_TOKEN` (required) — also accepts `HUGGINGFACEHUB_API_TOKEN` or `HF_API_KEY`
- `HF_PROVIDER` (optional, default: `novita`)
- `HF_MODEL` (optional, default: `zai-org/GLM-4.6`)

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

```bash
python news_agent_hf_toolcall.py
```

By default, the script prints “Daily Tech Headlines” for:

- `country="us"`
- `category="technology"`
- `limit=5`

## Programmatic usage

You can import and call `run_news_agent(...)`:

```python
from news_agent_hf_toolcall import run_news_agent

md = run_news_agent(country="us", category="technology", limit=5)
print(md)
```

Parameters:

- `country`: 2-letter country code (e.g. `"us"`)
- `category`: NewsAPI category (e.g. `"technology"`, `"business"`, `"sports"`)
- `limit`: number of headlines (int). The tool schema caps tool-called requests at **10**.

## How it works

### Tool: `fetch_top_headlines`

- Implemented in `fetch_top_headlines(country, category, limit)`
- Calls `https://newsapi.org/v2/top-headlines`
- Returns a trimmed JSON payload containing:
  - `title`
  - `description`
  - `url`
  - `source`

### Tool calling loop

`run_news_agent()`:

1. Sends a system + user prompt, plus `TOOLS` (OpenAI-style tool schema), to `InferenceClient.chat_completion(...)`.
2. If the model returns `tool_calls`, Python executes them and appends `role="tool"` messages containing the JSON output.
3. The model then generates the final Markdown summary.

### Fallback mode

If tool calling isn’t supported (or certain HTTP errors occur), it falls back to:

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
