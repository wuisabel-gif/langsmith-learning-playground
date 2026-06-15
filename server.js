import "dotenv/config";
import express from "express";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Client } from "langsmith";
import { runGraph, GRAPHS, setOpenAIKey } from "./graphs.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app = express();
app.use(express.json({ limit: "1mb" }));

const PORT = process.env.PORT || 3000;
const PROJECT = process.env.LANGSMITH_PROJECT || "agent-engineering-lab";

let hasOpenAI = !!process.env.OPENAI_API_KEY;
let hasLangSmith = !!process.env.LANGSMITH_API_KEY && process.env.LANGSMITH_TRACING !== "false";

let lsClient = hasLangSmith ? new Client() : null;

// Accept API keys entered in the UI and apply them to this running process
// (in-memory only — not written to disk). Lets you start without a .env file.
app.post("/api/keys", (req, res) => {
  const { openai, langsmith } = req.body || {};
  if (typeof openai === "string" && openai.trim()) {
    setOpenAIKey(openai.trim());
    hasOpenAI = true;
  }
  if (typeof langsmith === "string" && langsmith.trim()) {
    process.env.LANGSMITH_API_KEY = langsmith.trim();
    process.env.LANGSMITH_TRACING = "true";
    hasLangSmith = true;
    lsClient = new Client();
  }
  res.json({ openai: hasOpenAI, langsmith: hasLangSmith, model: process.env.OPENAI_MODEL || "gpt-4o-mini", project: PROJECT });
});

// ---- Config: lets the UI show real connection status ------------------------
app.get("/api/config", (req, res) => {
  res.json({
    openai: hasOpenAI,
    langsmith: hasLangSmith,
    model: process.env.OPENAI_MODEL || "gpt-4o-mini",
    project: PROJECT,
    graphs: Object.fromEntries(Object.entries(GRAPHS).map(([k, v]) => [k, v.name])),
  });
});

// ---- Run a real graph -------------------------------------------------------
app.post("/api/run", async (req, res) => {
  if (!hasOpenAI) {
    return res.status(400).json({ error: "OPENAI_API_KEY is not set. Add it to your .env file and restart the server." });
  }
  try {
    const { graph = "RetrievalGraph", input = "", attributes = {}, options = {} } = req.body || {};
    const result = await runGraph({ graph, input, attributes, options });
    res.json(result);
  } catch (e) {
    console.error("run error:", e);
    res.status(500).json({ error: e.message || "Run failed" });
  }
});

// ---- Read real traces back from LangSmith -----------------------------------
function runTypeToSpanType(rt) {
  const map = { llm: "llm", retriever: "retriever", tool: "tool", prompt: "prompt", parser: "parser", chain: "chain", evaluator: "evaluator" };
  return map[rt] || "chain";
}

app.get("/api/traces", async (req, res) => {
  if (!lsClient) return res.status(400).json({ error: "LangSmith is not configured (set LANGSMITH_API_KEY)." });
  try {
    const limit = Math.min(parseInt(req.query.limit) || 15, 50);
    const out = [];
    for await (const run of lsClient.listRuns({ projectName: PROJECT, isRoot: true, limit })) {
      const totalTokens = run.total_tokens || ((run.prompt_tokens || 0) + (run.completion_tokens || 0));
      out.push({
        id: run.id,
        name: run.name,
        timestamp: run.start_time ? new Date(run.start_time).toLocaleString() : "—",
        latency: run.end_time && run.start_time ? Math.round(new Date(run.end_time) - new Date(run.start_time)) : 0,
        tokens: totalTokens,
        cost: run.total_cost ? Number(run.total_cost) : 0,
        status: run.error ? "error" : "success",
        evalScore: run.feedback_stats ? 90 : (run.error ? 40 : 90),
        query: typeof run.inputs === "object" ? JSON.stringify(run.inputs).slice(0, 140) : String(run.inputs || ""),
        attributes: { topic: "general", ...(run.extra?.metadata || {}) },
        spans: [],
        live: true,
        fromLangSmith: true,
      });
    }
    res.json({ traces: out });
  } catch (e) {
    console.error("traces error:", e);
    res.status(500).json({ error: e.message });
  }
});

app.get("/api/traces/:id", async (req, res) => {
  if (!lsClient) return res.status(400).json({ error: "LangSmith is not configured." });
  try {
    const root = await lsClient.readRun(req.params.id);
    const children = [];
    for await (const child of lsClient.listRuns({ id: [], traceId: root.trace_id || root.id })) {
      if (child.id === root.id) continue;
      children.push(child);
    }
    children.sort((a, b) => new Date(a.start_time) - new Date(b.start_time));
    const t0 = new Date(root.start_time).getTime();
    const spans = children.map((c, i) => ({
      id: c.id,
      name: c.name,
      type: runTypeToSpanType(c.run_type),
      duration: c.end_time && c.start_time ? Math.round(new Date(c.end_time) - new Date(c.start_time)) : 0,
      startOffset: Math.max(0, Math.round(new Date(c.start_time).getTime() - t0)),
      status: c.error ? "error" : "success",
      tokens: c.total_tokens || 0,
      model: c.extra?.metadata?.ls_model_name || c.name,
      input: c.inputs || {},
      output: c.outputs || (c.error ? { error: c.error } : {}),
      conceptExplain: `Real ${c.run_type} run captured by LangSmith.`,
    }));
    res.json({
      id: root.id,
      name: root.name,
      latency: root.end_time ? Math.round(new Date(root.end_time) - new Date(root.start_time)) : 0,
      tokens: root.total_tokens || 0,
      cost: root.total_cost ? Number(root.total_cost) : 0,
      status: root.error ? "error" : "success",
      query: typeof root.inputs === "object" ? JSON.stringify(root.inputs) : String(root.inputs || ""),
      attributes: { topic: "general", ...(root.extra?.metadata || {}) },
      spans,
      fromLangSmith: true,
    });
  } catch (e) {
    console.error("trace detail error:", e);
    res.status(500).json({ error: e.message });
  }
});

// ---- Serve the UI -----------------------------------------------------------
app.get("/", (req, res) => res.sendFile(path.join(__dirname, "agent_engineering_lab.html")));
app.use(express.static(__dirname));

app.listen(PORT, () => {
  console.log(`\n  Agent Engineering Lab → http://localhost:${PORT}\n`);
  console.log(`  OpenAI:    ${hasOpenAI ? "✓ configured" : "✗ missing OPENAI_API_KEY"}`);
  console.log(`  LangSmith: ${hasLangSmith ? `✓ tracing to project "${PROJECT}"` : "✗ not tracing (set LANGSMITH_API_KEY)"}\n`);
});
