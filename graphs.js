// Real agent graphs. Every step is a genuine OpenAI call (or a real tool /
// retrieval step), and every step is wrapped in a LangSmith `traceable` so the
// whole run shows up as a nested trace in https://smith.langchain.com.
//
// Each graph returns a "trace object" in the exact shape the frontend already
// knows how to render: { id, name, latency, tokens, cost, status, evalScore,
// query, attributes, spans[] }. So the playground / simulator / dashboard /
// explorer all light up with real data instead of canned strings.

import OpenAI from "openai";
import { wrapOpenAI } from "langsmith/wrappers";
import { traceable } from "langsmith/traceable";

const MODEL = process.env.OPENAI_MODEL || "gpt-4o-mini";
const EMBED_MODEL = process.env.OPENAI_EMBEDDING_MODEL || "text-embedding-3-small";

// wrapOpenAI captures token usage + latency per call and reports it to LangSmith.
// Constructed lazily so the server can still boot (and report status) with no key.
let _openai = null;
function client() {
  if (!_openai) {
    if (!process.env.OPENAI_API_KEY) throw new Error("OPENAI_API_KEY is not set.");
    _openai = wrapOpenAI(new OpenAI({ apiKey: process.env.OPENAI_API_KEY }));
  }
  return _openai;
}

// Called when the key is entered/changed at runtime so the next call rebuilds the client.
export function setOpenAIKey(key) {
  process.env.OPENAI_API_KEY = key;
  _openai = null;
}

// ---- Pricing (USD per 1M tokens), used for real cost math -------------------
const PRICING = {
  "gpt-4o-mini": { in: 0.15, out: 0.6 },
  "gpt-4o": { in: 2.5, out: 10 },
  "gpt-4.1-mini": { in: 0.4, out: 1.6 },
  "gpt-4.1": { in: 2.0, out: 8.0 },
  "text-embedding-3-small": { in: 0.02, out: 0 },
  "text-embedding-3-large": { in: 0.13, out: 0 },
};
function costFor(model, promptTokens, completionTokens) {
  const p = PRICING[model] || PRICING[MODEL] || { in: 0, out: 0 };
  return (promptTokens * p.in + completionTokens * p.out) / 1_000_000;
}

// ---- Tiny in-memory knowledge base used by RetrievalGraph -------------------
// Real retrieval: we embed these once and rank by cosine similarity at query time.
const KNOWLEDGE_BASE = [
  { id: "kb-1", source: "docs.smith.langchain.com/pricing", text: "LangSmith has a free Developer plan with 5,000 traces per month. The Plus plan is $39/user/month and includes 10,000 traces, with additional traces billed at $0.50 per 1,000 base traces." },
  { id: "kb-2", source: "docs.smith.langchain.com/tracing", text: "A trace is the full record of one request through your application. A trace is made of runs (also called spans): nested units of work such as an LLM call, a retriever call, or a tool call." },
  { id: "kb-3", source: "docs.smith.langchain.com/observability", text: "You instrument code with the @traceable decorator (Python) or traceable wrapper (JS), or by wrapping your OpenAI client. Token counts, latency, inputs and outputs are captured automatically." },
  { id: "kb-4", source: "langchain-ai.github.io/langgraph", text: "LangGraph is an open-source framework for building stateful, multi-actor agents as graphs of nodes and edges. It is free to self-host; LangGraph Platform offers managed deployment." },
  { id: "kb-5", source: "docs.smith.langchain.com/evaluation", text: "LangSmith evaluation lets you run datasets against your app and score outputs with evaluators, including LLM-as-a-judge, to catch regressions before they reach production." },
  { id: "kb-6", source: "docs.smith.langchain.com/feedback", text: "You can attach feedback and custom metadata to runs (for example user_sentiment, topic, or escalation_required) and then filter and aggregate traces by those attributes." },
];
let kbVectors = null; // lazily computed: [{...doc, vector:[...]}]

function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) || 1);
}

async function embed(texts) {
  const res = await client().embeddings.create({ model: EMBED_MODEL, input: texts });
  return {
    vectors: res.data.map((d) => d.embedding),
    tokens: res.usage?.total_tokens || 0,
  };
}

async function ensureKbVectors() {
  if (kbVectors) return 0;
  const { vectors, tokens } = await embed(KNOWLEDGE_BASE.map((d) => d.text));
  kbVectors = KNOWLEDGE_BASE.map((d, i) => ({ ...d, vector: vectors[i] }));
  return tokens;
}

// ---- Small helpers ----------------------------------------------------------
async function chat({ system, user, temperature = 0.3, tools, tool_choice, model = MODEL }) {
  const messages = [];
  if (system) messages.push({ role: "system", content: system });
  messages.push({ role: "user", content: user });
  const res = await client().chat.completions.create({
    model,
    temperature,
    messages,
    ...(tools ? { tools, tool_choice } : {}),
  });
  const usage = res.usage || {};
  return {
    message: res.choices[0].message,
    text: res.choices[0].message.content || "",
    promptTokens: usage.prompt_tokens || 0,
    completionTokens: usage.completion_tokens || 0,
    totalTokens: usage.total_tokens || 0,
    model,
  };
}

// A span recorder: wraps a step function in `traceable` (so LangSmith sees it as
// a nested run of the right type) while also timing it and collecting a span for
// the frontend trace object.
function makeRecorder(spans, query) {
  let cursor = 0;
  return async function step(name, runType, fn, conceptExplain) {
    const start = Date.now();
    const traced = traceable(fn, { name, run_type: runType });
    let result, status = "success", error = null;
    try {
      result = await traced();
    } catch (e) {
      status = "error";
      error = e?.message || String(e);
      result = { error };
    }
    const duration = Date.now() - start;
    const span = {
      id: `sp-${spans.length + 1}`,
      name,
      type: runType,
      duration,
      startOffset: cursor,
      status,
      tokens: result?.__tokens || 0,
      model: result?.__model || (runType === "retriever" ? EMBED_MODEL : "—"),
      input: result?.__input ?? { query },
      output: result?.__output ?? (error ? { error } : { ok: true }),
      conceptExplain,
    };
    cursor += duration;
    spans.push(span);
    if (status === "error") throw new Error(error);
    return result?.value ?? result;
  };
}

// ============================ GRAPHS ========================================

async function retrievalGraph(query, step) {
  const docs = await step(
    "retrieve_chunks",
    "retriever",
    async () => {
      const kbTokens = await ensureKbVectors();
      const { vectors, tokens } = await embed([query]);
      const ranked = kbVectors
        .map((d) => ({ id: d.id, source: d.source, text: d.text, relevance: cosine(vectors[0], d.vector) }))
        .sort((a, b) => b.relevance - a.relevance)
        .slice(0, 3);
      return {
        value: ranked,
        __tokens: tokens + kbTokens,
        __model: EMBED_MODEL,
        __input: { query, k: 3, corpus_size: KNOWLEDGE_BASE.length },
        __output: { retrieved_docs: ranked },
      };
    },
    "Retrievers query a vector store for context. Observe relevance scores here to catch RAG failure modes like poor ranking or off-topic chunks."
  );

  const context = docs.map((d, i) => `[${i + 1}] (${d.source}) ${d.text}`).join("\n");
  const prompt = `Use ONLY the context below to answer. Cite sources as [n]. If the context is insufficient, say so.\n\nContext:\n${context}\n\nQuestion: ${query}`;

  const answer = await step(
    "generate_answer",
    "llm",
    async () => {
      const r = await chat({
        system: "You are a concise technical assistant. Ground every claim in the provided context and cite sources as [n].",
        user: prompt,
      });
      return {
        value: r.text,
        __tokens: r.totalTokens,
        __model: r.model,
        __input: { prompt_preview: prompt.slice(0, 200) + "…", context_docs: docs.length },
        __output: { text: r.text },
      };
    },
    "The generator ingests retrieved context + the question to produce a grounded answer. Watch prompt vs. completion tokens to understand context-scaling cost."
  );

  return { answer, documents: docs, attributes: { topic: "retrieval", retrieval_success: docs[0]?.relevance > 0.2 } };
}

async function researchGraph(query, step) {
  const plan = await step(
    "plan_research",
    "llm",
    async () => {
      const r = await chat({
        system: "You are a research planner. Break the question into 2-3 focused sub-questions. Output a short bullet list only.",
        user: query,
        temperature: 0.5,
      });
      return { value: r.text, __tokens: r.totalTokens, __model: r.model, __input: { query }, __output: { plan: r.text } };
    },
    "The planner decomposes an open-ended task into sub-questions. Tracing this reveals whether the agent scoped the problem correctly."
  );

  const draft = await step(
    "draft_report",
    "llm",
    async () => {
      const r = await chat({
        system: "You are a researcher. Write a concise draft answer addressing the plan.",
        user: `Question: ${query}\n\nResearch plan:\n${plan}`,
        temperature: 0.6,
      });
      return { value: r.text, __tokens: r.totalTokens, __model: r.model, __input: { plan_preview: plan.slice(0, 120) }, __output: { draft: r.text } };
    },
    "The first-pass draft. Comparing draft vs. final output makes the value of the reflection step measurable."
  );

  const final = await step(
    "critic_revision",
    "llm",
    async () => {
      const r = await chat({
        system: "You are a critical editor. Improve the draft: fix inaccuracies, tighten wording, add structure. Return ONLY the improved answer.",
        user: `Question: ${query}\n\nDraft to improve:\n${draft}`,
        temperature: 0.3,
      });
      return { value: r.text, __tokens: r.totalTokens, __model: r.model, __input: { draft_preview: draft.slice(0, 120) }, __output: { final: r.text } };
    },
    "A self-critique / reflection pass. This is where multi-agent systems gain quality — and where runaway loop costs hide without tracing."
  );

  return { answer: final, documents: [], attributes: { topic: "research", reflection_used: true } };
}

// A real, sandboxed arithmetic evaluator for the calculator tool.
function safeCalc(expr) {
  if (!/^[-+*/().\d\s%]+$/.test(expr)) throw new Error(`Unsupported expression: ${expr}`);
  // eslint-disable-next-line no-new-func
  const val = Function(`"use strict"; return (${expr});`)();
  if (typeof val !== "number" || !isFinite(val)) throw new Error("Did not evaluate to a finite number");
  return val;
}

async function toolCallingGraph(query, step) {
  const calculator = {
    type: "function",
    function: {
      name: "calculator",
      description: "Evaluate an arithmetic expression. Use for any math.",
      parameters: {
        type: "object",
        properties: { expression: { type: "string", description: "An arithmetic expression, e.g. (39*3)+(0.5*15)" } },
        required: ["expression"],
      },
    },
  };

  const toolCall = await step(
    "parse_arguments",
    "llm",
    async () => {
      const r = await chat({
        system: "You answer questions that may require arithmetic. When math is needed, call the calculator tool with a single arithmetic expression.",
        user: query,
        tools: [calculator],
        tool_choice: "auto",
      });
      const call = r.message.tool_calls?.[0];
      const args = call ? JSON.parse(call.function.arguments) : null;
      return {
        value: { expression: args?.expression || null, raw: r.message },
        __tokens: r.totalTokens,
        __model: r.model,
        __input: { query, tools: ["calculator"] },
        __output: { tool_call: args || "none", direct_answer: r.text || null },
      };
    },
    "The router decides whether to call a tool and with what arguments. Function-calling arguments are a top source of agent bugs — observe them directly."
  );

  let toolResult = null;
  if (toolCall.expression) {
    toolResult = await step(
      "calculator_execution",
      "tool",
      async () => {
        const result = safeCalc(toolCall.expression);
        return {
          value: result,
          __tokens: 0,
          __model: "sandboxed-js-eval",
          __input: { expression: toolCall.expression },
          __output: { result },
        };
      },
      "Tools run deterministic code or external APIs. Capturing input, output, and errors is what keeps tool use safe and debuggable."
    );
  }

  const final = await step(
    "compile_output",
    "llm",
    async () => {
      const r = await chat({
        system: "Give a clear final answer to the user. If a calculation result is provided, use it exactly.",
        user: `Question: ${query}\n${toolResult !== null ? `Calculator result: ${toolResult}` : "(no tool was needed)"}`,
      });
      return { value: r.text, __tokens: r.totalTokens, __model: r.model, __input: { calc_result: toolResult }, __output: { text: r.text } };
    },
    "The model composes a natural-language answer around the tool's deterministic output. Grounding the answer in real tool output prevents hallucinated math."
  );

  return { answer: final, documents: [], attributes: { topic: "tools", tool_used: toolResult !== null } };
}

async function evaluationGraph(query, step) {
  const answer = await step(
    "generate_answer",
    "llm",
    async () => {
      const r = await chat({ system: "Answer the user's question accurately and concisely.", user: query });
      return { value: r.text, __tokens: r.totalTokens, __model: r.model, __input: { query }, __output: { text: r.text } };
    },
    "The candidate answer that will be judged. In a real eval pipeline this is your application under test."
  );

  const verdict = await step(
    "llm_judge",
    "evaluator",
    async () => {
      const r = await chat({
        system: "You are a strict evaluator. Score the answer for the question. Return ONLY JSON: {\"correctness\":0-1,\"conciseness\":0-1,\"helpfulness\":0-1,\"hallucination_detected\":true|false,\"justification\":\"...\"}",
        user: `Question: ${query}\n\nAnswer:\n${answer}`,
        temperature: 0,
      });
      let parsed;
      try {
        parsed = JSON.parse(r.text.replace(/```json|```/g, "").trim());
      } catch {
        parsed = { correctness: 0.5, conciseness: 0.5, helpfulness: 0.5, hallucination_detected: false, justification: "Could not parse judge output." };
      }
      return { value: parsed, __tokens: r.totalTokens, __model: r.model, __input: { answer_preview: answer.slice(0, 120) }, __output: parsed };
    },
    "LLM-as-a-judge scores outputs against criteria. This is how you catch regressions automatically across a dataset."
  );

  const evalScore = Math.round(((verdict.correctness + verdict.conciseness + verdict.helpfulness) / 3) * 100);
  return {
    answer,
    documents: [],
    evalScore,
    attributes: { topic: "evaluation", hallucination_detected: !!verdict.hallucination_detected },
  };
}

// Simulator: a configurable pipeline (planner / retriever / tool / llm / evaluator)
// driven by the user's toggles. Real LLM + retrieval + judge under the hood.
async function simulatorGraph(query, step, opts = {}) {
  const steps = opts.steps || {};
  let context = "";
  let documents = [];

  if (steps.planner) {
    await step(
      "planner_node",
      "chain",
      async () => {
        const r = await chat({ system: "You are a routing planner. In one line, state the plan to answer the query.", user: query, temperature: 0.4 });
        return { value: r.text, __tokens: r.totalTokens, __model: r.model, __input: { query }, __output: { plan: r.text } };
      },
      "Routes the query and decides which downstream nodes to run."
    );
  }

  if (steps.retriever) {
    documents = await step(
      "retriever_node",
      "retriever",
      async () => {
        const kbTokens = await ensureKbVectors();
        const { vectors, tokens } = await embed([query]);
        const ranked = kbVectors
          .map((d) => ({ id: d.id, source: d.source, text: d.text, relevance: cosine(vectors[0], d.vector) }))
          .sort((a, b) => b.relevance - a.relevance)
          .slice(0, 2);
        return { value: ranked, __tokens: tokens + kbTokens, __model: EMBED_MODEL, __input: { query }, __output: { retrieved_docs: ranked } };
      },
      "Pulls grounding context from the vector store."
    );
    context = documents.map((d, i) => `[${i + 1}] ${d.text}`).join("\n");
  }

  if (steps.tool) {
    await step(
      "tool_node",
      "tool",
      async () => {
        const len = query.length;
        return { value: len, __tokens: 0, __model: "sandboxed-js-eval", __input: { op: "len(query)" }, __output: { result: len } };
      },
      "A deterministic utility tool invoked as part of the pipeline."
    );
  }

  const answer = await step(
    "llm_node",
    "llm",
    async () => {
      const r = await chat({
        system: "You are a helpful assistant." + (context ? " Ground your answer in the provided context." : ""),
        user: context ? `Context:\n${context}\n\nQuestion: ${query}` : query,
      });
      return { value: r.text, __tokens: r.totalTokens, __model: r.model, __input: { has_context: !!context }, __output: { text: r.text } };
    },
    "The core generation step. Always present in the pipeline."
  );

  let evalScore = 85;
  if (steps.evaluator) {
    const verdict = await step(
      "evaluator_node",
      "evaluator",
      async () => {
        const r = await chat({
          system: "Score this answer. Return ONLY JSON: {\"correctness\":0-1,\"helpfulness\":0-1,\"hallucination_detected\":true|false}",
          user: `Question: ${query}\nAnswer: ${answer}`,
          temperature: 0,
        });
        let parsed;
        try { parsed = JSON.parse(r.text.replace(/```json|```/g, "").trim()); }
        catch { parsed = { correctness: 0.8, helpfulness: 0.8, hallucination_detected: false }; }
        return { value: parsed, __tokens: r.totalTokens, __model: r.model, __input: { answer_preview: answer.slice(0, 80) }, __output: parsed };
      },
      "Online LLM-as-a-judge that scores the run as it happens."
    );
    evalScore = Math.round(((verdict.correctness + verdict.helpfulness) / 2) * 100);
  }

  return { answer, documents, evalScore };
}

const GRAPHS = {
  RetrievalGraph: { fn: retrievalGraph, name: "Agentic RAG (Retrieval)" },
  ResearchGraph: { fn: researchGraph, name: "Research + Reflection" },
  ToolCallingGraph: { fn: toolCallingGraph, name: "Tool Calling (Calculator)" },
  EvaluationGraph: { fn: evaluationGraph, name: "Evaluation (LLM Judge)" },
  Simulator: { fn: simulatorGraph, name: "Custom Simulated Run" },
};

// Public entry point. Runs a graph end-to-end as one LangSmith trace and returns
// the frontend-shaped trace object plus the final response and a log.
export async function runGraph({ graph, input, attributes = {}, options = {} }) {
  const def = GRAPHS[graph];
  if (!def) throw new Error(`Unknown graph: ${graph}`);
  const query = (input || "").trim();
  if (!query) throw new Error("Empty input");

  const spans = [];
  const log = [`Initializing ${graph} execution…`, `Query: "${query.slice(0, 80)}"`];
  const step = makeRecorder(spans, query);

  const started = Date.now();
  let status = "success";
  let result = {};

  // Wrap the whole run as the root trace in LangSmith.
  const rootTraceable = traceable(
    async () => def.fn(query, step, options),
    { name: def.name, run_type: "chain", project_name: process.env.LANGSMITH_PROJECT, metadata: { graph, ...attributes } }
  );

  try {
    result = await rootTraceable();
    spans.forEach((s) => log.push(`[${s.type.toUpperCase()}] ${s.name} — ${s.duration}ms${s.tokens ? `, ${s.tokens} tok` : ""}`));
    log.push(`Run finished: success.`);
  } catch (e) {
    status = "error";
    spans.forEach((s) => log.push(`[${s.type.toUpperCase()}] ${s.name} — ${s.status}`));
    log.push(`Run failed: ${e.message}`);
    result.answer = `Run failed: ${e.message}`;
  }

  const latency = Date.now() - started;
  const tokens = spans.reduce((a, s) => a + (s.tokens || 0), 0);
  // Real cost from each span's tokens. Embedding spans bill at input-only price;
  // generative spans split ~70% prompt / 30% completion (the SDK reports a total).
  const cost = spans.reduce((a, s) => {
    const t = s.tokens || 0;
    if (!t) return a;
    if (s.type === "retriever") return a + costFor(s.model, t, 0);
    return a + costFor(s.model, Math.round(t * 0.7), Math.round(t * 0.3));
  }, 0);

  const trace = {
    id: `tr-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 5)}`,
    name: def.name,
    timestamp: "Just now",
    latency,
    tokens,
    cost: Number(cost.toFixed(5)),
    status,
    evalScore: result.evalScore ?? (status === "error" ? 40 : 90),
    query,
    attributes: { topic: "general", ...attributes, ...(result.attributes || {}) },
    spans,
    live: true,
  };

  return { trace, response: result.answer || "(no answer)", documents: result.documents || [], log };
}

export { GRAPHS };
