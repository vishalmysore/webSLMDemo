/**
 * app.js — webSLMDemo main application
 *
 * Option 1 — Base model (user-selectable) + TF-IDF RAG (document-grounded)
 * Option 2 — Fine-tuned webSLM model or domain-prompted base model
 *
 * Two separate MLCEngine instances are used when different models are selected.
 * Queries run sequentially to stay within browser GPU memory limits.
 */

// Pin to 0.2.79 — the web-llm runtime our .wasm was compiled against
// (mlc-llm v0.19.0). A newer runtime can fail to instantiate the wasm (ABI drift).
import * as webllm from "https://esm.run/@mlc-ai/web-llm@0.2.79";
import { TFIDFRetriever } from "./rag.js";

// ── Base model list (Option 1 — RAG panel) ───────────────────────────────────
// IDs match WebLLM's built-in model registry. No appConfig needed.

const BASE_MODELS = [
  { id: 'Qwen2-0.5B-Instruct-q4f16_1-MLC',   label: 'Qwen2 0.5B ✓ recommended', size: '~400 MB', safe: true  },
  { id: 'Llama-3.2-1B-Instruct-q4f32_1-MLC', label: 'Llama 3.2 1B',             size: '~0.9 GB', safe: true  },
  { id: 'Qwen2.5-1.5B-Instruct-q4f16_1-MLC', label: 'Qwen2.5 1.5B',             size: '~1.1 GB', safe: true  },
  { id: 'Llama-3.2-3B-Instruct-q4f16_1-MLC', label: '⚠ Llama 3.2 3B',          size: '~2 GB',   safe: false },
];

// ── webSLM model (Option 2 — fine-tuned SLM panel) ──────────────────────────
// Only the compiled custom medical model. Right side is NOT user-selectable.

const WEBSLM_MODEL = {
  id:    'WebSLM-Custom-q4f16_1-webgpu',
  label: 'WebSLM-Medical-0.5B ✓ compiled',
  size:  '~293 MB',
  appConfig: {
    model_list: [{
      model:     'https://huggingface.co/VishalMysore/WebSLM-Custom-MLC',
      model_id:  'WebSLM-Custom-q4f16_1-webgpu',
      model_lib: 'https://huggingface.co/VishalMysore/WebSLM-Custom-MLC/resolve/main/libs/WebSLM-Custom-q4f16_1-webgpu.wasm',
    }],
  },
};

// ── Fine-tuning proof mode ──────────────────────────────────────────────────
// A controlled A/B: the SAME base our fine-tune started from (same q4f16_1 quant,
// a WebLLM prebuilt) vs the fine-tune itself. Identical system prompt (the one the
// model was actually TRAINED with) + greedy decoding → the only variable is the
// LoRA training, so any difference is attributable to fine-tuning.
const FINETUNE_BASE_MODEL_ID = 'Qwen2.5-0.5B-Instruct-q4f16_1-MLC';

// The system prompt from finetune/data/medical.jsonl — using it is what surfaces
// the fine-tuned behaviour (concise, plain language, "consult a professional").
const TRAINING_SYSTEM_PROMPT =
  "You are a careful medical information assistant. Provide general, educational " +
  "health information in plain language, and always recommend consulting a licensed " +
  "healthcare professional for diagnosis or treatment.";

// ✓ trained = present in the training set (strongest signal); the rest are held-out
// (test whether the trained STYLE generalises to unseen questions).
const PROOF_EXAMPLES = [
  { q: "What are common signs of dehydration?",   trained: true  },
  { q: "Do antibiotics treat the flu?",           trained: true  },
  { q: "When is chest pain a medical emergency?", trained: true  },
  { q: "How should I use ibuprofen safely?",      trained: false },
];

const DOMAIN_EXAMPLES = {
  insurance: [
    "What does comprehensive auto coverage include?",
    "Is water damage covered under a standard homeowners policy?",
    "What is the difference between term and whole life insurance?",
    "How does a health insurance deductible work?",
  ],
  medical: [
    "What are the symptoms of type 2 diabetes?",
    "When should I seek emergency care for chest pain?",
    "What are common side effects of statins?",
    "How should I use ibuprofen safely?",
  ],
  legal: [
    "What is the statute of limitations for personal injury?",
    "Can my employer legally monitor my work emails?",
    "What rights do I have if my landlord won't make repairs?",
    "What makes a contract legally enforceable?",
  ],
};

const DOMAIN_SYSTEM_PROMPTS = {
  insurance: `You are a knowledgeable insurance specialist assistant trained specifically for the insurance domain. You help users understand policies, coverage types, claims processes, premiums, and insurance terminology with precision and clarity. Always use proper insurance terminology and provide accurate, detailed explanations. Include relevant caveats such as state-specific variations and always remind users to consult a licensed insurance professional for binding advice. Be professional, empathetic, and thorough.`,
  medical: `You are a medical information specialist trained specifically for healthcare questions. You provide accurate, evidence-based health information to help users understand symptoms, conditions, medications, and treatments. Always use appropriate medical terminology while keeping explanations accessible. Consistently emphasize that your responses are general health information, not a substitute for professional medical advice, and that users must consult qualified healthcare providers for diagnosis and treatment decisions.`,
  legal: `You are a legal information specialist trained specifically for legal questions. You help users understand legal concepts, their rights, legal processes, and relevant statutes with precision. Use proper legal terminology and cite relevant principles. Always clarify that your responses constitute general legal information only and are not legal advice. Users must consult a qualified attorney for advice specific to their situation. Be precise, balanced, and clear about legal complexity.`,
};

// ── State ─────────────────────────────────────────────────────────────────────

let engine    = null;   // Base model engine (Option 1 RAG)
let engineSLM = null;   // webSLM engine (Option 2); null means reuse engine
let retriever = null;
let knowledgeBase = {};
let currentDomain = "insurance";
let isGenerating  = false;
let baseReady  = false;
let webslmReady = false;
let mode = "product";        // "product" (Base+RAG vs SLM) | "proof" (Base vs Fine-tune)
let baseLoadedId = null;     // which model id the base engine currently holds

// ── DOM refs ──────────────────────────────────────────────────────────────────

const $ = id => document.getElementById(id);
const queryInput   = $("queryInput");
const sendBtn      = $("sendBtn");
const domainSelect = $("domainSelect");
const statusRAG    = $("statusRAG");
const statusSLM    = $("statusSLM");
const progressRAG  = $("progressRAG");
const progressSLM  = $("progressSLM");
const historyRAG   = $("historyRAG");
const historySLM   = $("historySLM");
const retrievedDocs = $("retrievedDocs");
const exampleBtns   = $("exampleBtns");
const settingsToggle = $("settingsToggle");
const settingsPanel  = $("settingsPanel");
const modeToggle     = $("modeToggle");
const examplesLabel  = $("examplesLabel");
const badgeRAG       = $("badgeRAG");
const badgeSLM       = $("badgeSLM");
const panelTitleRAG  = $("panelTitleRAG");
const panelSubRAG    = $("panelSubRAG");
const panelTitleSLM  = $("panelTitleSLM");
const panelSubSLM    = $("panelSubSLM");
// Base model controls
const baseModelSelect  = $("baseModelSelect");
const loadBaseBtn      = $("loadBaseBtn");
const baseModelNote    = $("baseModelNote");
const baseModelProg    = $("baseModelProg");
// webSLM model controls
const webslmModelSelect = $("webslmModelSelect");
const loadWebslmBtn     = $("loadWebslmBtn");
const webslmModelNote   = $("webslmModelNote");
const webslmModelProg   = $("webslmModelProg");

// ── Init ──────────────────────────────────────────────────────────────────────

async function init() {
  // WebGPU is required to RUN models, but not to explore the UI (mode toggle,
  // examples, settings). Warn + disable loading, but don't bail — keep the page usable.
  const webgpuOK = !!navigator.gpu;
  if (!webgpuOK) {
    const msg = "WebGPU not supported — use Chrome 113+, Edge 113+, or Firefox Nightly with WebGPU enabled.";
    statusRAG.textContent = msg;
    statusSLM.textContent = msg;
    statusRAG.style.color = "#ef4444";
    statusSLM.style.color = "#ef4444";
  }

  // Load knowledge base
  const res = await fetch("./knowledge-base.json");
  knowledgeBase = await res.json();
  retriever = new TFIDFRetriever(knowledgeBase[currentDomain] || []);

  // Populate base model dropdown
  BASE_MODELS.forEach(m => {
    const opt = document.createElement("option");
    opt.value = m.id;
    opt.textContent = `${m.label} (${m.size})`;
    baseModelSelect.appendChild(opt);
  });
  baseModelSelect.disabled = !webgpuOK;
  updateBaseModelNote();

  // webSLM model is fixed (no dropdown) — display note only

  // Render example queries
  renderExamples(currentDomain);

  // Settings toggle
  settingsToggle.addEventListener("click", () =>
    settingsPanel.classList.toggle("open")
  );

  // Mode toggle (Product demo ↔ Fine-tuning proof)
  modeToggle.querySelectorAll("button").forEach(btn =>
    btn.addEventListener("click", () => setMode(btn.dataset.mode))
  );

  // Open settings on first load so users see the model selectors
  settingsPanel.classList.add("open");

  // Base model selector change
  baseModelSelect.addEventListener("change", () => {
    if (engine) {
      engine.unload?.();
      engine = null;
      baseReady = false;
      baseLoadedId = null;
      statusRAG.textContent = "Model changed — click Load to reload.";
      progressRAG.style.width = "0%";
      updateSendBtn();
    }
    updateBaseModelNote();
  });

  // webSLM model is fixed — no change event listener needed

  loadBaseBtn.addEventListener("click", loadBaseModel);
  loadWebslmBtn.addEventListener("click", loadWebslmModel);
  loadBaseBtn.disabled   = !webgpuOK;
  loadWebslmBtn.disabled = !webgpuOK;

  domainSelect.addEventListener("change", e => {
    currentDomain = e.target.value;
    retriever = new TFIDFRetriever(knowledgeBase[currentDomain] || []);
    renderExamples(currentDomain);
    clearPanels();
  });

  sendBtn.addEventListener("click", handleSend);
  queryInput.addEventListener("keydown", e => {
    if (e.key === "Enter" && !e.shiftKey) handleSend();
  });
}

function updateSendBtn() {
  sendBtn.disabled = !(baseReady && webslmReady) || isGenerating;
}

function updateBaseModelNote() {
  const m = BASE_MODELS.find(x => x.id === baseModelSelect.value);
  if (!m) return;
  baseModelNote.textContent = m.safe ? "" : `⚠ ${m.size} — needs a dedicated GPU with enough VRAM. If it crashes, switch to Qwen2 0.5B.`;
  baseModelNote.style.color = m.safe ? "" : "#f59e0b";
}

// webSLM model note is static (no dropdown selection)

// Remove chat bubbles from both panels but PRESERVE the #retrievedDocs container
// (it lives inside #historyRAG; innerHTML="" would detach the cached reference).
function clearPanels() {
  historyRAG.querySelectorAll(".chat-message").forEach(n => n.remove());
  historySLM.querySelectorAll(".chat-message").forEach(n => n.remove());
  retrievedDocs.innerHTML = "";
}

// Switch between "product" (Base+RAG vs Fine-tuned SLM) and "proof"
// (identical base vs fine-tune, same training prompt, greedy decoding).
function setMode(m) {
  if (m === mode) return;
  mode = m;
  const proof = m === "proof";

  modeToggle.querySelectorAll("button").forEach(b =>
    b.classList.toggle("active", b.dataset.mode === m));

  // RAG controls (domain + retrieved docs) only apply to product mode.
  domainSelect.style.display = proof ? "none" : "";

  if (proof) {
    badgeRAG.textContent = "Base";
    badgeSLM.textContent = "Fine-tuned";
    panelTitleRAG.textContent = "Qwen2.5-0.5B-Instruct — base";
    panelSubRAG.textContent   = "Same base • NO fine-tuning • same training prompt • greedy";
    panelTitleSLM.textContent = "WebSLM-Medical-0.5B — fine-tuned";
    panelSubSLM.textContent   = "Same base + your LoRA training • same prompt • greedy";
    examplesLabel.textContent = "Try (✓ = seen in training):";
    baseModelNote.textContent = "Proof mode: base is fixed to Qwen2.5-0.5B-Instruct — the exact base your fine-tune started from.";
    baseModelNote.style.color = "";
    renderProofExamples();
  } else {
    badgeRAG.textContent = "Option 1";
    badgeSLM.textContent = "Option 2";
    panelTitleRAG.textContent = "Base + RAG";
    panelSubRAG.textContent   = "General model • Documents retrieved and injected into context";
    panelTitleSLM.textContent = "Fine-tuned webSLM";
    panelSubSLM.textContent   = "Domain-specialized model • No retrieval • Behaviour baked in during training";
    examplesLabel.textContent = "Try:";
    updateBaseModelNote();
    renderExamples(currentDomain);
  }

  // The BASE model differs between modes; if the loaded one no longer matches what
  // this mode needs, drop it and require a reload so we never compare the wrong base.
  const neededBase = proof ? FINETUNE_BASE_MODEL_ID : baseModelSelect.value;
  if (baseLoadedId !== neededBase) {
    if (engine) { engine.unload?.(); engine = null; }
    baseReady = false;
    baseLoadedId = null;
    progressRAG.style.width = "0%";
    statusRAG.style.color = "";
    statusRAG.textContent = proof
      ? "Proof mode — click Load Base Model (loads Qwen2.5-0.5B base)."
      : "Mode changed — click Load Base Model.";
  }

  clearPanels();
  updateSendBtn();
}

function renderProofExamples() {
  exampleBtns.innerHTML = PROOF_EXAMPLES
    .map(e => `<button class="example-btn" data-q="${e.q}">${e.trained ? "✓ " : ""}${e.q}</button>`)
    .join("");
  exampleBtns.querySelectorAll(".example-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      queryInput.value = btn.dataset.q;
      queryInput.focus();
    });
  });
}

async function loadBaseModel() {
  const proof = mode === "proof";
  // Proof mode forces the exact base the fine-tune started from (apples-to-apples).
  const modelId = proof ? FINETUNE_BASE_MODEL_ID : baseModelSelect.value;
  const label = proof
    ? "Qwen2.5-0.5B-Instruct (base)"
    : (BASE_MODELS.find(m => m.id === modelId) || BASE_MODELS[0]).label;
  loadBaseBtn.disabled = true;
  baseReady = false;
  updateSendBtn();
  statusRAG.textContent = `Loading ${label}…`;
  progressRAG.style.width = "0%";
  statusRAG.style.color = "";

  const onProgress = report => {
    statusRAG.textContent = report.text;
    progressRAG.style.width = Math.round((report.progress || 0) * 100) + "%";
    baseModelProg.textContent = report.text;
  };

  try {
    if (engine) { engine.unload?.(); engine = null; }
    engine = new webllm.MLCEngine({ initProgressCallback: onProgress });
    await engine.reload(modelId);
    baseLoadedId = modelId;
    statusRAG.textContent = proof ? `✓ ${label}` : `✓ ${label} + RAG`;
    progressRAG.style.width = "100%";
    baseModelProg.textContent = "";
    baseReady = true;
    updateSendBtn();
  } catch (err) {
    statusRAG.textContent = `Failed: ${err.message}`;
    statusRAG.style.color = "#ef4444";
    baseModelProg.textContent = "";
    console.error(err);
  } finally {
    loadBaseBtn.disabled = false;
  }
}

async function loadWebslmModel() {
  // Load the fixed compiled webSLM medical model
  const wm = WEBSLM_MODEL;

  loadWebslmBtn.disabled = true;
  webslmReady = false;
  updateSendBtn();
  statusSLM.textContent = `Loading ${wm.label}…`;
  progressSLM.style.width = "0%";
  statusSLM.style.color = "";

  const onProgress = report => {
    statusSLM.textContent = report.text;
    progressSLM.style.width = Math.round((report.progress || 0) * 100) + "%";
    webslmModelProg.textContent = report.text;
  };

  try {
    if (engineSLM) { engineSLM.unload?.(); engineSLM = null; }
    // appConfig + initProgressCallback go in the ENGINE constructor.
    // reload()'s 2nd arg is ChatOptions and has no appConfig field — passing it
    // there is silently ignored, so the engine never learns about the custom
    // model and throws "Cannot find model record in appConfig for …".
    engineSLM = new webllm.MLCEngine({
      appConfig: wm.appConfig,
      initProgressCallback: onProgress,
    });
    await engineSLM.reload(wm.appConfig.model_list[0].model_id);
    statusSLM.textContent = `✓ ${wm.label}`;
    progressSLM.style.width = "100%";
    webslmModelProg.textContent = "";
    webslmReady = true;
    updateSendBtn();
  } catch (err) {
    statusSLM.textContent = `Failed: ${err.message}`;
    statusSLM.style.color = "#ef4444";
    webslmModelProg.textContent = "";
    console.error(err);
  } finally {
    loadWebslmBtn.disabled = false;
  }
}

// ── Query handling ────────────────────────────────────────────────────────────

async function handleSend() {
  const query = queryInput.value.trim();
  if (!query || isGenerating || !baseReady) return;

  isGenerating = true;
  sendBtn.disabled = true;
  queryInput.value = "";

  // Show user turn in both panels
  appendMessage(historyRAG, "user", query);
  appendMessage(historySLM, "user", query);

  let leftMessages, rightMessages, rightEngine, opts;

  if (mode === "proof") {
    // Controlled A/B: BOTH sides get the identical training system prompt, no RAG,
    // and greedy decoding. The only variable left is the LoRA fine-tuning, so any
    // difference between the panels is attributable to training.
    retrievedDocs.innerHTML = "";
    const msgs = [
      { role: "system", content: TRAINING_SYSTEM_PROMPT },
      { role: "user",   content: query },
    ];
    leftMessages  = msgs;          // base Qwen2.5-0.5B
    rightMessages = msgs;          // fine-tune
    rightEngine   = engineSLM;     // must be the fine-tune — never fall back to base here
    // Identical decoding for BOTH panels (so the only variable is the fine-tuning).
    // Use Qwen2.5's OWN recommended sampling (temp 0.7 / top_p 0.8, straight from its
    // mlc-chat-config). NOTE: temp 0 made these 0.5B models loop, and adding strong
    // frequency/presence penalties made the bilingual base drift into Chinese gibberish
    // — normal sampling diversity is the stable choice and needs no penalties.
    opts = { temperature: 0.7, top_p: 0.8, max_tokens: 256 };
  } else {
    // Product demo: base + RAG (left) vs domain-prompted fine-tune (right).
    const docs = retriever.retrieve(query, 3);
    renderRetrievedDocs(docs);
    const contextBlock = docs.length > 0
      ? `Relevant reference material:\n\n${docs.map(d =>
          `[${d.doc.title}]\n${d.doc.content}`
        ).join("\n\n")}\n\n---\nUsing the above material where relevant, answer the following question:`
      : null;
    leftMessages = [
      { role: "system", content: "You are a helpful general-purpose assistant. When reference material is provided, use it to give accurate and grounded answers. If no relevant material is available, answer based on general knowledge." },
      { role: "user",   content: contextBlock ? `${contextBlock}\n\n${query}` : query },
    ];
    rightMessages = [
      { role: "system", content: DOMAIN_SYSTEM_PROMPTS[currentDomain] },
      { role: "user",   content: query },
    ];
    rightEngine = engineSLM || engine;
    opts = { temperature: 0.7, top_p: 0.95, max_tokens: 512 };
  }

  // Show thinking indicators
  const ragThinking = appendThinking(historyRAG);
  const slmThinking = appendThinking(historySLM);

  // Left runs first (shared GPU; sequential to stay within browser memory limits).
  await streamResponse(engine, leftMessages, historyRAG, ragThinking, opts);
  await streamResponse(rightEngine, rightMessages, historySLM, slmThinking, opts);

  isGenerating = false;
  sendBtn.disabled = false;
  updateSendBtn();
}

async function streamResponse(eng, messages, historyEl, thinkingEl, opts = {}) {
  const { temperature = 0.7, top_p = 0.95, max_tokens = 512,
          frequency_penalty = 0, presence_penalty = 0 } = opts;
  try {
    const stream = await eng.chat.completions.create({
      messages,
      stream: true,
      temperature,
      top_p,
      max_tokens,
      frequency_penalty,
      presence_penalty,
    });

    thinkingEl.remove();
    const msgDiv = appendMessage(historyEl, "assistant", "");
    const bubble = msgDiv.querySelector(".chat-bubble");

    let fullText = "";
    for await (const chunk of stream) {
      const delta = chunk.choices[0]?.delta?.content || "";
      fullText += delta;
      bubble.textContent = fullText;
      historyEl.scrollTop = historyEl.scrollHeight;
    }
  } catch (err) {
    thinkingEl.remove();
    appendMessage(historyEl, "assistant", `⚠ Error: ${err.message}`);
    console.error(err);
  }
}

// ── UI helpers ────────────────────────────────────────────────────────────────

function appendMessage(container, role, text) {
  const div = document.createElement("div");
  div.className = `chat-message ${role}`;
  const bubble = document.createElement("div");
  bubble.className = "chat-bubble";
  bubble.textContent = text;
  div.appendChild(bubble);
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
  return div;
}

function appendThinking(container) {
  const div = document.createElement("div");
  div.className = "chat-message assistant";
  div.innerHTML = `<div class="chat-bubble thinking-bubble">
    <span class="dot"></span><span class="dot"></span><span class="dot"></span>
  </div>`;
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
  return div;
}

function renderRetrievedDocs(docs) {
  if (docs.length === 0) {
    retrievedDocs.innerHTML = `<p class="no-docs">No relevant documents found — answering from general knowledge.</p>`;
    return;
  }
  retrievedDocs.innerHTML = `
    <div class="retrieved-section">
      <h4 class="retrieved-heading">Retrieved Context (${docs.length} doc${docs.length > 1 ? "s" : ""})</h4>
      ${docs.map(d => `
        <div class="doc-chip">
          <div class="doc-chip-header">
            <span class="doc-title">${d.doc.title}</span>
            <span class="doc-score">relevance: ${(d.score * 100).toFixed(0)}%</span>
          </div>
          <p class="doc-snippet">${d.doc.content.slice(0, 130)}…</p>
        </div>
      `).join("")}
    </div>`;
}

function renderExamples(domain) {
  const examples = DOMAIN_EXAMPLES[domain] || [];
  exampleBtns.innerHTML = examples
    .map(e => `<button class="example-btn">${e}</button>`)
    .join("");
  exampleBtns.querySelectorAll(".example-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      queryInput.value = btn.textContent;
      queryInput.focus();
    });
  });
}

// ── Start ─────────────────────────────────────────────────────────────────────

init().catch(console.error);
