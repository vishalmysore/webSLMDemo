/**
 * app.js — webSLMDemo main application
 *
 * Uses a SINGLE WebLLM engine to compare:
 *   Option 1 — Base Qwen2.5-1.5B + TF-IDF RAG (document-grounded)
 *   Option 2 — Same base model with domain-tuned system prompt
 *              (or a custom compiled webSLM model if a config URL is provided)
 *
 * Queries run sequentially through one engine to stay within browser GPU memory.
 */

import * as webllm from "https://esm.run/@mlc-ai/web-llm";
import { TFIDFRetriever } from "./rag.js";

// ── Configuration ────────────────────────────────────────────────────────────

const BASE_MODEL_ID = "Qwen2.5-1.5B-Instruct-q4f16_1-MLC";

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

let engine = null;         // Shared WebLLM engine (base model)
let engineSLM = null;      // Optional second engine for custom fine-tuned model
let retriever = null;
let knowledgeBase = {};
let currentDomain = "insurance";
let isGenerating = false;

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
const exampleBtns  = $("exampleBtns");
const settingsToggle = $("settingsToggle");
const settingsPanel  = $("settingsPanel");
const customModelUrl = $("customModelUrl");
const loadCustomBtn  = $("loadCustomBtn");

// ── Init ──────────────────────────────────────────────────────────────────────

async function init() {
  // Check WebGPU availability
  if (!navigator.gpu) {
    const msg = "WebGPU is not supported in this browser. Please use Chrome 113+, Edge 113+, or Firefox Nightly with WebGPU enabled.";
    statusRAG.textContent = msg;
    statusSLM.textContent = msg;
    statusRAG.style.color = "#ef4444";
    statusSLM.style.color = "#ef4444";
    return;
  }

  // Load knowledge base
  const res = await fetch("./knowledge-base.json");
  knowledgeBase = await res.json();
  retriever = new TFIDFRetriever(knowledgeBase[currentDomain] || []);

  // Render example queries
  renderExamples(currentDomain);

  // Wire up controls
  settingsToggle.addEventListener("click", () =>
    settingsPanel.classList.toggle("open")
  );

  loadCustomBtn.addEventListener("click", loadCustomModel);

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

  // Load base model (used for both panels unless custom model provided)
  await loadBaseModel();
}

async function loadBaseModel() {
  const onProgress = report => {
    const pct = Math.round((report.progress || 0) * 100);
    statusRAG.textContent = report.text;
    statusSLM.textContent = report.text;
    progressRAG.style.width = pct + "%";
    progressSLM.style.width = pct + "%";
  };

  try {
    statusRAG.textContent = "Loading Qwen2.5-1.5B…";
    statusSLM.textContent = "Loading Qwen2.5-1.5B…";
    engine = new webllm.MLCEngine();
    await engine.reload(BASE_MODEL_ID, { initProgressCallback: onProgress });
    statusRAG.textContent = "✓ Ready — Qwen2.5-1.5B + RAG";
    statusSLM.textContent = "✓ Ready — Qwen2.5-1.5B (domain-tuned prompt)";
    progressRAG.style.width = "100%";
    progressSLM.style.width = "100%";
    sendBtn.disabled = false;
  } catch (err) {
    const msg = `Failed to load model: ${err.message}`;
    statusRAG.textContent = msg;
    statusSLM.textContent = msg;
    statusRAG.style.color = "#ef4444";
    statusSLM.style.color = "#ef4444";
    console.error(err);
  }
}

async function loadCustomModel() {
  const url = customModelUrl.value.trim();
  if (!url) return;
  statusSLM.textContent = "Loading custom webSLM model…";
  progressSLM.style.width = "0%";
  try {
    const configRes = await fetch(url);
    if (!configRes.ok) throw new Error(`HTTP ${configRes.status}`);
    const modelConfig = await configRes.json();
    engineSLM = new webllm.MLCEngine();
    await engineSLM.reload(modelConfig, {
      initProgressCallback: report => {
        statusSLM.textContent = report.text;
        progressSLM.style.width = Math.round((report.progress || 0) * 100) + "%";
      },
    });
    statusSLM.textContent = "✓ Ready — Custom webSLM model";
    progressSLM.style.width = "100%";
  } catch (err) {
    statusSLM.textContent = `Custom model failed: ${err.message}`;
    statusSLM.style.color = "#ef4444";
    engineSLM = null;
  }
}

// ── Query handling ────────────────────────────────────────────────────────────

async function handleSend() {
  const query = queryInput.value.trim();
  if (!query || isGenerating || !engine) return;

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
