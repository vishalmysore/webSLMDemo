# webSLMDemo

Live demo comparing two approaches to domain-specific AI in the browser — both running 100% on-device via WebGPU with no server required.

**[Live Demo](https://vishalmysore.github.io/webSLMDemo)**

---

## What it shows

| | Option 1 — Base + RAG | Option 2 — Fine-tuned webSLM |
|---|---|---|
| **Model** | Qwen2.5-1.5B (base) | Compiled webSLM model (or domain-prompted base) |
| **Knowledge** | Retrieved at query time via TF-IDF | Baked into model weights during fine-tuning |
| **Context** | Retrieved docs injected into prompt | Domain system prompt only |
| **Visible retrieval** | Yes — shows matched docs + scores | No retrieval step |

The side-by-side view makes clear when RAG helps (factual grounding) and where fine-tuning differs (tone, domain authority, appropriate disclaimers without explicit context).

---

## Running locally

No build step needed. Serve the directory with any static file server:

```bash
# Python
python -m http.server 8080

# Node
npx serve .
```

Then open `http://localhost:8080` in Chrome 113+ or Edge 113+ (WebGPU required).

**Note:** The first load downloads the Qwen2.5-1.5B model (~1 GB). Subsequent loads use the browser cache.

---

## Using your own webSLM fine-tuned model

1. Build your model using the [webSLM pipeline](https://github.com/vishalmysore/webSLM)
2. Host the compiled artifacts (weight shards + `.wasm`) on GitHub Releases or a CDN
3. In the demo, click **⚙ Settings** and paste the URL to your `mlc-chat-config.json`
4. Click **Load Model** — the right panel will use your fine-tuned weights

---

## Project structure

```
index.html          Main comparison UI
app.js              WebLLM integration and query orchestration
rag.js              In-browser TF-IDF retriever (no dependencies)
knowledge-base.json Domain document store (insurance, medical, legal)
.github/workflows/  GitHub Pages deployment
```

---

## Related

- [webSLM](https://github.com/vishalmysore/webSLM) — pipeline for building and deploying domain SLMs to the browser
- [WebLLM](https://github.com/mlc-ai/web-llm) — in-browser LLM inference via WebGPU
- [MLC-LLM](https://github.com/mlc-ai/mlc-llm) — model compilation toolchain
