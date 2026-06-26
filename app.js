/**
 * app.js — webSLMDemo main application
 *
 * Option 1 — Base model (user-selectable) + TF-IDF RAG (document-grounded)
 * Option 2 — Fine-tuned webSLM model or domain-prompted base model
 *
 * Two separate MLCEngine instances are used when different models are selected.
 * Queries run sequentially to stay within browser GPU memory limits.
 */

import * as webllm from "https://esm.run/@mlc-ai/web-llm";
import { TFIDFRetriever } from "./rag.js";

// ── Base model list (Option 1 — RAG panel) ───────────────────────────────────
// IDs match WebLLM's built-in model registry. No appConfig needed.

const BASE_MODELS = [
  { id: 'Qwen2-0.5B-Instruct-q4f16_1-MLC',   label: 'Qwen2 0.5B ✓ recommended', size: '~400 MB', safe: true  },
  { id: 'Llama-3.2-1B-Instruct-q4f32_1-MLC', label: 'Llama 3.2 1B',             size: '~0.9 GB', safe: true  },
  { id: 'Qwen2.5-1.5B-Instruct-q4f16_1-MLC', label: 'Qwen2.5 1.5B',             size: '~1.1 GB', safe: true  },
  { id: 'Llama-3.2-3B-Instruct-q4f16_1-MLC', label: '⚠ Llama 3.2 3B',          size: '~2 GB',   safe: false },
];

// ── webSLM model list (Option 2 — fine-tuned SLM panel) ──────────────────────
// Custom models use appConfig to provide the HF model URL + compiled .wasm lib.
// Models marked needsCompilation=true cannot be loaded yet; a note is shown.

const WEBSLM_MODELS = [
  {
    id:    'Qwen2-0.5B-Instruct-q4f16_1-MLC',
    label: 'Qwen2 0.5B (domain-prompted base)',
    size:  '~400 MB',
    appConfig: null,
    needsCompilation: false,
  },
  {
    id:    'WebSLM-Medical-0.5B-MLC',
    label: 'WebSLM-Medical-0.5B ✓ compiled',
    size:  '~293 MB',
    appConfig: {
      model_list: [{
        model:     'https://huggingface.co/VishalMysore/WebSLM-Custom-MLC',
        model_id:  'WebSLM-Medical-0.5B-MLC',
        model_lib: 'https://huggingface.co/VishalMysore/WebSLM-Custom-MLC/resolve/main/libs/WebSLM-Custom-q4f16_1-webgpu.wasm',
      }],
    },
    needsCompilation: false,
  },
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
  // Check WebGPU availability
  if (!navigator.gpu) {
    const msg = "WebGPU not supported — use Chrome 113+, Edge 113+, or Firefox Nightly with WebGPU enabled.";
    statusRAG.textContent = msg;
    statusSLM.textContent = msg;
    statusRAG.style.color = "#ef4444";
    statusSLM.style.color = "#ef4444";
    loadBaseBtn.disabled  = true;
    loadWebslmBtn.disabled = true;
    return;
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
  baseModelSelect.disabled = false;
  updateBaseModelNote();

  // Populate webSLM model dropdown
  WEBSLM_MODELS.forEach(m => {
    const opt = document.createElement("option");
    opt.value = m.id;
    opt.textContent = `${m.label} (${m.size})`;
    webslmModelSelect.appendChild(opt);
  });
  webslmModelSelect.disabled = false;
  updateWebslmModelNote();

  // Render example queries
  renderExamples(currentDomain);

  // Settings toggle
  settingsToggle.addEventListener("click", () =>
    settingsPanel.classList.toggle("open")
  );

  // Open settings on first load so users see the model selectors
  settingsPanel.classList.add("open");

  // Base model selector change
  baseModelSelect.addEventListener("change", () => {
    if (engine) {
      engine.unload?.();
      engine = null;
      baseReady = false;
      statusRAG.textContent = "Model changed — click Load to reload.";
      progressRAG.style.width = "0%";
      updateSendBtn();
    }
    updateBaseModelNote();
  });

  // webSLM model selector change
  webslmModelSelect.addEventListener("change", () => {
    if (engineSLM) {
      engineSLM.unload?.();
      engineSLM = null;
      webslmReady = false;
      statusSLM.textContent = "Model changed — click Load to reload.";
      progressSLM.style.width = "0%";
      updateSendBtn();
    }
    updateWebslmModelNote();
  });

  loadBaseBtn.addEventListener("click", loadBaseModel);
  loadWebslmBtn.addEventListener("click", loadWebslmModel);

  domainSelect.addEventListener("change", e => {
    currentDomain = e.target.value;
    retriever = new TFIDFRetriever(knowledgeBase[currentDomain] || []);
    renderExamples(currentDomain);
    retrievedDocs.innerHTML = "";
    historyRAG.innerHTML = "";
    historySLM.innerHTML = "";
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

function updateWebslmModelNote() {
  const m = WEBSLM_MODELS.find(x => x.id === webslmModelSelect.value);
  if (!m) return;
  if (m.needsCompilation) {
    webslmModelNote.textContent = m.compilationNote;
    webslmModelNote.style.color = "#f59e0b";
    loadWebslmBtn.disabled = true;
  } else {
    webslmModelNote.textContent = m.appConfig
      ? `Custom compiled webSLM model — loads from HuggingFace (VishalMysore/WebSLM-Custom-MLC).`
      : `Base model with domain-specialized system prompt. No separate download — shares base engine.`;
    webslmModelNote.style.color = "#8b949e";
    loadWebslmBtn.disabled = false;
  }
}

async function loadBaseModel() {
  const modelId = baseModelSelect.value;
  const modelMeta = BASE_MODELS.find(m => m.id === modelId) || BASE_MODELS[0];
  loadBaseBtn.disabled = true;
  baseReady = false;
  updateSendBtn();
  statusRAG.textContent = `Loading ${modelMeta.label}…`;
  progressRAG.style.width = "0%";
  statusRAG.style.color = "";

  const onProgress = report => {
    statusRAG.textContent = report.text;
    progressRAG.style.width = Math.round((report.progress || 0) * 100) + "%";
    baseModelProg.textContent = report.text;
  };

  try {
    if (engine) { engine.unload?.(); engine = null; }
    engine = new webllm.MLCEngine();
    await engine.reload(modelId, { initProgressCallback: onProgress });
    statusRAG.textContent = `✓ ${modelMeta.label} + RAG`;
    progressRAG.style.width = "100%";
    baseModelProg.textContent = "";
    baseReady = true;

    // If webSLM model is the same built-in model, mark it ready too (shared engine)
    const wm = WEBSLM_MODELS.find(x => x.id === webslmModelSelect.value);
    if (wm && !wm.needsCompilation && !wm.appConfig && wm.id === modelId) {
      engineSLM = null;
      webslmReady = true;
      statusSLM.textContent = `✓ ${modelMeta.label} (domain-tuned prompt)`;
      progressSLM.style.width = "100%";
    }
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
  const wm = WEBSLM_MODELS.find(x => x.id === webslmModelSelect.value);
  if (!wm || wm.needsCompilation) return;

  // Same built-in model as base — share the engine, no extra download
  if (!wm.appConfig) {
    if (!baseReady) {
      webslmModelNote.textContent = "Load the base model first.";
      webslmModelNote.style.color = "#f59e0b";
      return;
    }
    engineSLM = null;
    webslmReady = true;
    const bm = BASE_MODELS.find(m => m.id === baseModelSelect.value) || BASE_MODELS[0];
    statusSLM.textContent = `✓ ${bm.label} (domain-tuned prompt)`;
    progressSLM.style.width = "100%";
    updateSendBtn();
    return;
  }

  // Custom compiled webSLM model — load with appConfig
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
    engineSLM = new webllm.MLCEngine();
    await engineSLM.reload(wm.id, {
      appConfig: wm.appConfig,
      initProgressCallback: onProgress,
    });
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

  // Retrieve relevant docs for Option 1
  const docs = retriever.retrieve(query, 3);
  renderRetrievedDocs(docs);

  // Build Option 1 messages (base + RAG context)
  const contextBlock = docs.length > 0
    ? `Relevant reference material:\n\n${docs.map(d =>
        `[${d.doc.title}]\n${d.doc.content}`
      ).join("\n\n")}\n\n---\nUsing the above material where relevant, answer the following question:`
    : null;

  const ragMessages = [
    {
      role: "system",
      content: "You are a helpful general-purpose assistant. When reference material is provided, use it to give accurate and grounded answers. If no relevant material is available, answer based on general knowledge.",
    },
    {
      role: "user",
      content: contextBlock ? `${contextBlock}\n\n${query}` : query,
    },
  ];

  // Build Option 2 messages (domain-specialized prompt, no retrieval)
  const slmMessages = [
    { role: "system", content: DOMAIN_SYSTEM_PROMPTS[currentDomain] },
    { role: "user", content: query },
  ];

  // Show thinking indicators
  const ragThinking = appendThinking(historyRAG);
  const slmThinking = appendThinking(historySLM);

  // Option 1 — Base + RAG (runs first on the shared engine)
  await streamResponse(engine, ragMessages, historyRAG, ragThinking);

  // Option 2 — Fine-tuned SLM or domain-prompted base
  const activeEngine = engineSLM || engine;
  await streamResponse(activeEngine, slmMessages, historySLM, slmThinking);

  isGenerating = false;
  sendBtn.disabled = false;
  updateSendBtn();
}

async function streamResponse(eng, messages, historyEl, thinkingEl) {
  try {
    const stream = await eng.chat.completions.create({
      messages,
      stream: true,
      temperature: 0.7,
      max_tokens: 512,
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
