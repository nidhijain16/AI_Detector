import { useState, useRef, useMemo } from "react";

// ═══════════════════════════════════════════════════════
//  API ROUTES (Local FastAPI Server)
// ═══════════════════════════════════════════════════════
const API_BASE = "http://localhost:8000/api";

// ═══════════════════════════════════════════════════════
//  CLIENT-SIDE SIGNAL COMPUTATION  (no API needed)
// ═══════════════════════════════════════════════════════

const AI_TRANSITIONS = [
  "furthermore", "moreover", "additionally", "in conclusion", "in summary",
  "it is important to note", "it is worth noting", "notably", "significantly",
  "consequently", "it should be noted", "in addition", "ultimately", "therefore",
  "thus", "needless to say", "to summarize", "to conclude", "as a result",
  "in the realm of", "in today's world", "delve into", "underscore", "crucial",
  "on the other hand", "it goes without saying", "as we can see"
];

function splitSentences(text) {
  return (text.match(/[^.!?\n]+[.!?\n]+/g) || [])
    .map(s => s.trim())
    .filter(s => s.split(/\s+/).length >= 3);
}

/** Burstiness = coefficient of variation of sentence lengths.
 *  Human: CV > 0.45 (varied). AI: CV < 0.25 (uniform). */
function computeBurstiness(text) {
  const sentences = splitSentences(text);
  if (sentences.length < 2) return { cv: 0.3, score: 50, label: "Need more text", sentences, lengths: [] };
  const lengths = sentences.map(s => s.split(/\s+/).length);
  const mean = lengths.reduce((a, b) => a + b, 0) / lengths.length;
  const variance = lengths.reduce((a, b) => a + (b - mean) ** 2, 0) / lengths.length;
  const stdDev = Math.sqrt(variance);
  const cv = mean > 0 ? stdDev / mean : 0;
  // 0=AI-like, 100=human-like (normalized: cv=0 → 0, cv≥0.6 → 100)
  const humanScore = Math.round(Math.min(cv / 0.6, 1) * 100);
  const label = cv < 0.2 ? "Very Uniform (AI-like)" : cv < 0.35 ? "Moderately Varied" : cv < 0.5 ? "Varied" : "Highly Varied (Human-like)";
  return { cv: +cv.toFixed(3), score: humanScore, mean: +mean.toFixed(1), stdDev: +stdDev.toFixed(1), label, sentences, lengths };
}

/** Type-Token Ratio: unique_words / total_words.
 *  Low TTR = repetitive vocabulary = AI signal. */
function computeTTR(text) {
  const words = (text.toLowerCase().match(/\b[a-z]{2,}\b/g) || []);
  if (!words.length) return { ttr: 0.5, aiScore: 50 };
  const unique = new Set(words);
  const ttr = unique.size / words.length;
  // TTR < 0.35 very AI-like, > 0.6 very human
  const aiScore = Math.round(Math.max(0, Math.min(1, (0.65 - ttr) / 0.4)) * 100);
  return { ttr: +ttr.toFixed(3), unique: unique.size, total: words.length, aiScore };
}

/** Transition word density — AI loves formal connectors. */
function computeTransitions(text) {
  const lower = text.toLowerCase();
  const found = AI_TRANSITIONS.filter(t => lower.includes(t));
  const wordCount = (text.match(/\b\w+\b/g) || []).length;
  const density = wordCount > 0 ? (found.length / wordCount) * 100 : 0;
  const aiScore = Math.min(Math.round(density * 35), 100);
  return { found, count: found.length, density: +density.toFixed(2), aiScore };
}

/** Trigram repetition — AI text reuses 3-word phrases more. */
function computeRepetition(text) {
  const words = (text.toLowerCase().match(/\b[a-z]+\b/g) || []);
  if (words.length < 6) return { ratio: 0, aiScore: 0 };
  const trigrams = {};
  for (let i = 0; i < words.length - 2; i++) {
    const k = `${words[i]} ${words[i + 1]} ${words[i + 2]}`;
    trigrams[k] = (trigrams[k] || 0) + 1;
  }
  const total = Object.keys(trigrams).length;
  const repeated = Object.values(trigrams).filter(v => v > 1).length;
  const ratio = total > 0 ? repeated / total : 0;
  return { ratio: +ratio.toFixed(3), aiScore: Math.min(Math.round(ratio * 250), 100) };
}

/** Average sentence similarity — AI sentences follow similar patterns. */
function computeSentenceUniformity(text) {
  const sentences = splitSentences(text);
  if (sentences.length < 3) return { score: 0, aiScore: 50 };
  // Use sentence-length patterns as proxy
  const lengths = sentences.map(s => s.split(/\s+/).length);
  const max = Math.max(...lengths), min = Math.min(...lengths);
  const range = max - min;
  // Very small range = uniform = AI
  const uniformityAI = Math.round(Math.max(0, Math.min(1, (30 - range) / 25)) * 100);
  return { range, max, min, aiScore: uniformityAI };
}

/** Master function: combines all local signals. */
function computeLocalSignals(text) {
  const burstiness = computeBurstiness(text);
  const ttr = computeTTR(text);
  const transitions = computeTransitions(text);
  const repetition = computeRepetition(text);
  const uniformity = computeSentenceUniformity(text);

  // Weighted combination → local AI probability
  const localAI = Math.round(
    (1 - burstiness.score / 100) * 100 * 0.30 +
    ttr.aiScore * 0.25 +
    transitions.aiScore * 0.20 +
    repetition.aiScore * 0.15 +
    uniformity.aiScore * 0.10
  );

  return { burstiness, ttr, transitions, repetition, uniformity, localAI };
}

// ═══════════════════════════════════════════════════════
//  PLAGIARISM  (Jaccard similarity on word n-grams, client-side)
// ═══════════════════════════════════════════════════════

function jaccardSim(a, b) {
  const sA = new Set(a.toLowerCase().match(/\b\w+\b/g) || []);
  const sB = new Set(b.toLowerCase().match(/\b\w+\b/g) || []);
  if (!sA.size || !sB.size) return 0;
  const inter = [...sA].filter(x => sB.has(x)).length;
  return inter / (sA.size + sB.size - inter);
}

const STOCK_PHRASES = [
  "in today's fast-paced world", "in this day and age", "since the dawn of time",
  "it goes without saying", "needless to say", "last but not least",
  "at the end of the day", "think outside the box", "paradigm shift",
  "going forward", "circle back", "synergy", "leverage", "deep dive",
  "take it to the next level", "move the needle", "in the realm of",
  "shed light on", "unpack the complexities"
];

function computePlagiarismSignals(text) {
  const sentences = splitSentences(text);
  const lower = text.toLowerCase();

  const pairs = [];
  for (let i = 0; i < sentences.length; i++) {
    for (let j = i + 1; j < sentences.length; j++) {
      const sim = jaccardSim(sentences[i], sentences[j]);
      if (sim > 0.40) pairs.push({ i, j, sim, sA: sentences[i], sB: sentences[j] });
    }
  }

  const foundStock = STOCK_PHRASES.filter(p => lower.includes(p));
  const highSim = pairs.filter(p => p.sim > 0.65);

  const penalty = (highSim.length / Math.max(sentences.length, 1)) * 45
    + foundStock.length * 4;
  const originality = Math.max(0, Math.min(100, Math.round(100 - penalty)));

  const verdict =
    originality >= 85 ? "Highly Original" :
      originality >= 65 ? "Mostly Original" :
        originality >= 45 ? "Partially Repetitive" :
          originality >= 25 ? "Low Originality" : "Highly Repetitive";

  return {
    sentences, pairs: pairs.sort((a, b) => b.sim - a.sim).slice(0, 8),
    foundStock, originality, verdict
  };
}

// ═══════════════════════════════════════════════════════
//  LOCAL API CALLS
// ═══════════════════════════════════════════════════════

async function uploadToBackend(file, signal) {
  const formData = new FormData();
  formData.append("file", file);
  const resp = await fetch(`${API_BASE}/upload_and_analyze`, {
    method: "POST", body: formData, signal
  });
  if (!resp.ok) {
    const e = await resp.json().catch(() => ({}));
    throw new Error(e.detail || e.message || `API error ${resp.status}`);
  }
  return resp.json();
}

async function analyzeTextToBackend(text, signal) {
  const file = new File([text], "pasted.txt", { type: "text/plain" });
  return uploadToBackend(file, signal);
}

function chunkTextSimple(text, maxWords = 300) {
  const sentences = splitSentences(text);
  const chunks = [];
  let cur = "";
  for (const s of sentences) {
    if ((cur.split(" ").length + s.split(" ").length) > maxWords && cur) {
      chunks.push(cur.trim());
      cur = s + " ";
    } else cur += s + " ";
  }
  if (cur.trim()) chunks.push(cur.trim());
  return chunks;
}

// ═══════════════════════════════════════════════════════
//  UI COMPONENTS
// ═══════════════════════════════════════════════════════

const C = {
  bg: "#06090f",
  surface: "#0c1220",
  border: "#1a2540",
  borderLight: "#243050",
  text: "#dce8f0",
  muted: "#4a6080",
  dim: "#253245",
  amber: "#f59e0b",
  teal: "#2dd4bf",
  red: "#f87171",
  green: "#4ade80",
  blue: "#60a5fa",
};

function GaugeBar({ label, value, aiScore, invert = false, helpText }) {
  // aiScore: 0=human-like, 100=AI-like
  const fillColor = aiScore > 65 ? C.red : aiScore > 35 ? C.amber : C.teal;
  const humanScore = 100 - aiScore;
  return (
    <div style={{ marginBottom: 14 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 5, alignItems: "baseline" }}>
        <span style={{ fontSize: 11, fontFamily: "monospace", color: C.muted, letterSpacing: "0.08em" }}>{label}</span>
        <span style={{ fontSize: 11, fontFamily: "monospace", color: fillColor }}>
          {aiScore > 65 ? "AI-like" : aiScore > 35 ? "Mixed" : "Human-like"} · {value}
        </span>
      </div>
      <div style={{ height: 6, background: C.dim, borderRadius: 3, overflow: "hidden" }}>
        <div style={{ height: "100%", width: `${aiScore}%`, background: fillColor, borderRadius: 3, transition: "width 0.8s ease" }} />
      </div>
      {helpText && <div style={{ fontSize: 10, color: C.dim, marginTop: 3, fontFamily: "monospace" }}>{helpText}</div>}
    </div>
  );
}

function ScoreRing({ score, label, mode = "ai", size = 100 }) {
  const r = (size / 2) - 10;
  const circ = 2 * Math.PI * r;
  const displayScore = mode === "ai" ? score : score; // both 0-100
  const offset = circ - (displayScore / 100) * circ;
  const color = mode === "ai"
    ? (score > 65 ? C.red : score > 35 ? C.amber : C.teal)
    : (score > 65 ? C.teal : score > 35 ? C.amber : C.red);
  const cx = size / 2, cy = size / 2;
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        <circle cx={cx} cy={cy} r={r} fill="none" stroke={C.dim} strokeWidth="8" />
        <circle cx={cx} cy={cy} r={r} fill="none" stroke={color} strokeWidth="8"
          strokeDasharray={circ} strokeDashoffset={offset} strokeLinecap="round"
          transform={`rotate(-90 ${cx} ${cy})`}
          style={{ transition: "stroke-dashoffset 1.2s cubic-bezier(.4,0,.2,1)" }} />
        <text x={cx} y={cy - 4} textAnchor="middle" fill={C.text} fontSize={size * 0.19} fontWeight="700" fontFamily="monospace">{score}%</text>
        <text x={cx} y={cy + size * 0.12} textAnchor="middle" fill={C.muted} fontSize={size * 0.09} fontFamily="monospace">{label}</text>
      </svg>
    </div>
  );
}

function BurstinessChart({ lengths, mean }) {
  if (!lengths?.length) return null;
  const maxLen = Math.max(...lengths, 1);
  return (
    <div style={{ marginTop: 10 }}>
      <div style={{ fontSize: 10, color: C.muted, fontFamily: "monospace", marginBottom: 6, letterSpacing: "0.08em" }}>
        SENTENCE LENGTHS (words per sentence) — mean: {mean}w
      </div>
      <div style={{ display: "flex", alignItems: "flex-end", gap: 3, height: 40 }}>
        {lengths.map((l, i) => {
          const h = Math.max(4, (l / maxLen) * 36);
          const deviation = Math.abs(l - parseFloat(mean));
          const varied = deviation > parseFloat(mean) * 0.3;
          return (
            <div key={i} title={`Sentence ${i + 1}: ${l} words`}
              style={{ flex: 1, height: h, background: varied ? C.teal : C.amber, borderRadius: "2px 2px 0 0", opacity: 0.85, transition: "height 0.5s ease", cursor: "default" }} />
          );
        })}
      </div>
      <div style={{ display: "flex", gap: 12, marginTop: 6, fontSize: 10, color: C.muted, fontFamily: "monospace" }}>
        <span style={{ color: C.teal }}>■</span> varied (human-like)
        <span style={{ color: C.amber }}>■</span> uniform (AI-like)
      </div>
    </div>
  );
}

function SentenceHighlight({ sentences, burstiness }) {
  if (!sentences?.length) return null;
  const { lengths, mean } = burstiness;
  return (
    <div style={{ marginTop: 12 }}>
      <div style={{ fontSize: 10, color: C.muted, fontFamily: "monospace", marginBottom: 8, letterSpacing: "0.08em" }}>
        PER-SENTENCE ANALYSIS
      </div>
      <div style={{ fontSize: 13, lineHeight: 1.9, color: C.text }}>
        {sentences.map((s, i) => {
          const len = lengths[i] || s.split(/\s+/).length;
          const deviation = Math.abs(len - parseFloat(mean));
          const isVaried = deviation > parseFloat(mean) * 0.3;
          const hasTransition = AI_TRANSITIONS.some(t => s.toLowerCase().includes(t));
          const borderColor = hasTransition ? C.red : isVaried ? C.teal : C.amber;
          return (
            <span key={i} title={`${len} words${hasTransition ? " · AI transition detected" : ""}`}
              style={{
                borderBottom: `2px solid ${borderColor}`,
                paddingBottom: 1, marginRight: 4,
                backgroundColor: hasTransition ? "rgba(248,113,113,0.08)" : "transparent",
                borderRadius: 2,
                cursor: "help",
              }}>
              {s}
            </span>
          );
        })}
      </div>
      <div style={{ display: "flex", gap: 14, marginTop: 8, fontSize: 10, color: C.muted, fontFamily: "monospace" }}>
        <span><span style={{ color: C.teal }}>—</span> varied length</span>
        <span><span style={{ color: C.amber }}>—</span> uniform length</span>
        <span><span style={{ color: C.red }}>—</span> AI transition word</span>
      </div>
    </div>
  );
}

function SimilarityHeatmap({ sentences, pairs }) {
  if (!sentences?.length) return null;
  const n = Math.min(sentences.length, 12);
  const matrix = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => {
      if (i === j) return 1;
      const p = pairs.find(p => (p.i === i && p.j === j) || (p.i === j && p.j === i));
      return p ? p.sim : 0;
    })
  );
  const cellSize = Math.max(20, Math.min(38, Math.floor(360 / n)));
  return (
    <div style={{ overflowX: "auto" }}>
      <div style={{ fontSize: 10, color: C.muted, fontFamily: "monospace", marginBottom: 8, letterSpacing: "0.08em" }}>
        SENTENCE SIMILARITY MATRIX (Jaccard) — {n} sentences shown
      </div>
      <div style={{ display: "inline-block" }}>
        {matrix.map((row, i) => (
          <div key={i} style={{ display: "flex" }}>
            {row.map((val, j) => {
              const bg = i === j
                ? C.blue
                : val > 0.6 ? `rgba(248,113,113,${val})`
                  : val > 0.3 ? `rgba(245,158,11,${val + 0.1})`
                    : `rgba(45,212,191,${val * 0.3})`;
              return (
                <div key={j} title={`S${i + 1} vs S${j + 1}: ${(val * 100).toFixed(0)}% similar`}
                  style={{ width: cellSize, height: cellSize, background: bg, border: `1px solid ${C.bg}`, cursor: "default" }} />
              );
            })}
          </div>
        ))}
      </div>
      <div style={{ display: "flex", gap: 12, marginTop: 8, fontSize: 10, color: C.muted, fontFamily: "monospace" }}>
        <span><span style={{ color: C.red }}>■</span> high similarity (&gt;60%)</span>
        <span><span style={{ color: C.amber }}>■</span> moderate (30-60%)</span>
        <span><span style={{ color: C.blue }}>■</span> self</span>
      </div>
    </div>
  );
}

function Pill({ text, color = C.muted }) {
  return (
    <span style={{
      background: `${color}18`, border: `1px solid ${color}55`, color,
      padding: "2px 8px", borderRadius: 3, fontSize: 11, fontFamily: "monospace",
      display: "inline-block", margin: "2px"
    }}>{text}</span>
  );
}

function VerdictBanner({ verdict, score, mode = "ai" }) {
  const isAI = mode === "ai";
  const isGood = isAI ? score < 35 : score > 65;
  const isMid = isAI ? (score >= 35 && score <= 65) : (score >= 35 && score <= 65);
  const isBad = isAI ? score > 65 : score < 35;
  const color = isBad ? C.red : isMid ? C.amber : C.teal;
  return (
    <div style={{
      background: `${color}0f`, border: `1px solid ${color}40`,
      borderRadius: 8, padding: "14px 18px",
      display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 12
    }}>
      <div>
        <div style={{ fontSize: 10, color: C.muted, fontFamily: "monospace", letterSpacing: "0.1em", marginBottom: 4 }}>
          {mode === "ai" ? "AI DETECTION VERDICT" : "ORIGINALITY VERDICT"}
        </div>
        <div style={{ fontSize: 20, fontWeight: 700, color, fontFamily: "monospace" }}>{verdict}</div>
      </div>
      <ScoreRing score={score} label={mode === "ai" ? "AI PROB" : "ORIGINAL"} mode={mode} size={90} />
    </div>
  );
}

function InfoRow({ label, value, note }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", padding: "7px 0", borderBottom: `1px solid ${C.border}` }}>
      <span style={{ fontSize: 11, color: C.muted, fontFamily: "monospace", letterSpacing: "0.06em" }}>{label}</span>
      <span style={{ fontSize: 12, color: C.text, fontFamily: "monospace", textAlign: "right" }}>
        {value}
        {note && <span style={{ color: C.muted, marginLeft: 6, fontSize: 10 }}>{note}</span>}
      </span>
    </div>
  );
}

function Spinner({ size = 14 }) {
  return <div style={{ width: size, height: size, border: `2px solid ${C.amber}`, borderTopColor: "transparent", borderRadius: "50%", animation: "spin 0.7s linear infinite", display: "inline-block" }} />;
}

// ═══════════════════════════════════════════════════════
//  CONFIG PANEL
// ═══════════════════════════════════════════════════════

function ConfigPanel({ open, setOpen }) {
  return (
    <div style={{ position: "relative" }}>
      <button onClick={() => setOpen(!open)} style={{
        background: C.surface, border: `1px solid ${C.border}`, color: C.muted,
        padding: "6px 14px", borderRadius: 5, cursor: "pointer",
        fontSize: 11, fontFamily: "monospace", letterSpacing: "0.08em"
      }}>⚙ CONFIG</button>
      {open && (
        <div style={{
          position: "absolute", right: 0, top: "calc(100% + 8px)", zIndex: 200,
          background: "#0a1020", border: `1px solid ${C.borderLight}`, borderRadius: 10,
          padding: 20, width: 360, boxShadow: "0 24px 64px rgba(0,0,0,0.8)"
        }}>
          <div style={{ fontSize: 11, color: C.muted, fontFamily: "monospace", letterSpacing: "0.1em", marginBottom: 16 }}>LOCAL FASTAPI SERVER</div>
          <div style={{ background: C.bg, borderRadius: 6, padding: "12px 14px", fontSize: 11, fontFamily: "monospace", lineHeight: 2 }}>
            <div style={{ color: C.muted, marginBottom: 8, letterSpacing: "0.08em" }}>MODELS USED:</div>
            <div><span style={{ color: C.amber }}>Detector</span> <span style={{ color: C.dim }}>→</span> <span style={{ color: C.blue }}>roberta-base-openai-detector</span></div>
            <div style={{ color: C.dim, fontSize: 10, marginBottom: 6 }}>Runs entirely locally via Python Transformers</div>
            <div><span style={{ color: C.amber }}>Humanizer</span> <span style={{ color: C.dim }}>→</span> <span style={{ color: C.blue }}>Vamsi/T5_Paraphrase_Paws</span></div>
            <div style={{ color: C.dim, fontSize: 10, marginBottom: 6 }}>Streams chunks in real-time.</div>
            <div><span style={{ color: C.amber }}>API Connection</span> <span style={{ color: C.dim }}>→</span> <span style={{ color: C.teal }}>localhost:8000</span></div>
          </div>
          <div style={{ marginTop: 12, background: "#061f14", border: `1px solid #059669`, borderRadius: 5, padding: "10px 12px", fontSize: 10, color: "#10b981", fontFamily: "monospace", lineHeight: 1.8 }}>
            ✓ 100+ Page Documents supported via Python chunking engine.
          </div>
        </div>
      )}
    </div>
  );
}

// ═══════════════════════════════════════════════════════
//  MAIN APP
// ═══════════════════════════════════════════════════════

export default function App() {
  const [tab, setTab] = useState(0);
  const [text, setText] = useState("");
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState("");
  const [configOpen, setConfigOpen] = useState(false);

  const [detectResult, setDetectResult] = useState(null);
  const [humanizeResult, setHumanizeResult] = useState(null);
  const [plagResult, setPlagResult] = useState(null);
  const [copied, setCopied] = useState(false);

  const abortRef = useRef(null);

  // Live preview of local signals while typing
  const liveSignals = useMemo(
    () => text.trim().length > 80 ? computeLocalSignals(text) : null,
    [text]
  );

  const abort = () => { abortRef.current?.abort(); setLoading(false); };

  // ── DETECT ──────────────────────────────────────────
  const handleDetect = async () => {
    if (!text.trim() && !file) return;
    setLoading(true); setError(""); setDetectResult(null); setProgress(0);
    const ab = new AbortController(); abortRef.current = ab;
    try {
      setProgress(20);
      const res = await (file ? uploadToBackend(file, ab.signal) : analyzeTextToBackend(text, ab.signal));

      let analysisText = text;
      if (res.full_text && res.full_text !== text) {
        analysisText = res.full_text;
        setText(analysisText); // show the parsed document in textarea
      }

      setProgress(80);
      const local = computeLocalSignals(analysisText);
      const robertaScore = res.overall_ai_score; // Given by our Python backend

      const finalAI = robertaScore !== undefined && robertaScore > 0
        ? Math.round(robertaScore * 0.65 + local.localAI * 0.35)
        : local.localAI;

      const verdict =
        finalAI >= 80 ? "AI Generated" :
          finalAI >= 62 ? "Likely AI" :
            finalAI >= 40 ? "Mixed / Uncertain" :
              finalAI >= 22 ? "Likely Human" : "Human Written";

      setDetectResult({ finalAI, robertaScore, local, verdict, usedModel: robertaScore !== undefined && robertaScore > 0 });
      setProgress(100);
    } catch (e) { if (e.name !== "AbortError") setError(e.message); }
    finally { setLoading(false); }
  };

  // ── HUMANIZE (Streaming from Backend) ─────────────────────────────
  const handleHumanize = async () => {
    if (!text.trim() && !file) return;
    setLoading(true); setError(""); setHumanizeResult(null); setProgress(0);
    const ab = new AbortController(); abortRef.current = ab;

    // First Ensure we have the text string to chunk
    let baseText = text;
    if (file && !text) {
      try {
        setProgress(5);
        const parseRes = await uploadToBackend(file, ab.signal);
        baseText = parseRes.full_text;
        setText(baseText);
      } catch (e) { setError("Failed reading file"); setLoading(false); return; }
    }

    try {
      const before = computeLocalSignals(baseText);
      const chunks = chunkTextSimple(baseText, 250); // Safe token window for T5

      // Start SSE Streaming Response from FastAPI
      const response = await fetch(`${API_BASE}/humanize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ chunks }),
        signal: ab.signal
      });

      if (!response.body) throw new Error("No response body");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let done = false;
      let humanizedOutput = [];
      let completionCount = 0;

      while (!done) {
        const { value, done: streamDone } = await reader.read();
        done = streamDone;
        if (value) {
          const chunkStr = decoder.decode(value, { stream: true });
          const messages = chunkStr.split('\\n\\n').filter(s => s.trim().startsWith('data:'));

          for (let msg of messages) {
            const dataStr = msg.replace('data: ', '').trim();
            if (dataStr === '[DONE]') break;
            try {
              const data = JSON.parse(dataStr);
              humanizedOutput[data.chunk_id] = data.rewritten_text;
              completionCount++;
              setProgress(Math.round((completionCount / chunks.length) * 100));
            } catch (err) { }
          }
        }
      }

      const finalHumanizedString = humanizedOutput.join(" ");
      const after = computeLocalSignals(finalHumanizedString);
      setHumanizeResult({
        original: baseText,
        humanized: finalHumanizedString,
        before,
        after,
        iterations: 1,
        finalAI: after.localAI
      });
      setProgress(100);
    } catch (e) { if (e.name !== "AbortError") setError(e.message); }
    finally { setLoading(false); }
  };

  // ── PLAGIARISM ──────────────────────────────────────
  const handlePlagiarism = () => {
    if (!text.trim()) return;
    setLoading(true); setError(""); setPlagResult(null);
    setTimeout(() => {
      setPlagResult(computePlagiarismSignals(text));
      setLoading(false);
    }, 200);
  };

  const actions = [handleDetect, handleHumanize, handlePlagiarism];
  const actionLabels = ["RUN DETECTOR", "HUMANIZE", "CHECK PLAGIARISM"];
  const wordCount = text.trim() ? text.trim().split(/\s+/).length : 0;

  return (
    <div onClick={() => configOpen && setConfigOpen(false)} style={{ minHeight: "100vh", background: C.bg, color: C.text, fontFamily: "'IBM Plex Sans', 'Segoe UI', sans-serif", paddingBottom: 80 }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        textarea, input, select { font-family: inherit; }
        textarea:focus, input:focus { outline: none !important; border-color: ${C.amber} !important; }
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes fadeUp { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes shimmer { 0%,100% { opacity: 0.6; } 50% { opacity: 1; } }
        .panel { animation: fadeUp 0.35s ease forwards; }
        ::-webkit-scrollbar { width: 4px; height: 4px; }
        ::-webkit-scrollbar-track { background: ${C.bg}; }
        ::-webkit-scrollbar-thumb { background: ${C.border}; border-radius: 2px; }
      `}</style>

      {/* ── HEADER ── */}
      <div style={{ background: C.surface, borderBottom: `1px solid ${C.border}`, padding: "0 20px", position: "sticky", top: 0, zIndex: 100 }}>
        <div style={{ maxWidth: 860, margin: "0 auto" }}>
          <div style={{ padding: "16px 0 14px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <div style={{ width: 32, height: 32, background: `linear-gradient(135deg, #1a3a5c, #0f2040)`, border: `1px solid ${C.borderLight}`, borderRadius: 6, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 16 }}>🧬</div>
              <div>
                <div style={{ fontFamily: "monospace", fontWeight: 700, fontSize: 14, letterSpacing: "0.05em", color: C.text }}>
                  FORENSIC<span style={{ color: C.amber }}>TEXT</span>
                </div>
                <div style={{ fontSize: 9, color: C.muted, fontFamily: "monospace", letterSpacing: "0.12em" }}>
                  RoBERTa · T5 · Jaccard · Burstiness
                </div>
              </div>
            </div>
            <div onClick={e => e.stopPropagation()}>
              <ConfigPanel open={configOpen} setOpen={setConfigOpen} />
            </div>
          </div>

          {/* Tabs */}
          <div style={{ display: "flex", gap: 2 }}>
            {[
              { label: "AI DETECTOR", icon: "⚡" },
              { label: "HUMANIZER", icon: "↺" },
              { label: "PLAGIARISM", icon: "⊕" },
            ].map(({ label, icon }, i) => (
              <button key={i} onClick={() => { setTab(i); setError(""); }} style={{
                background: "none", border: "none", cursor: "pointer",
                padding: "9px 16px", fontSize: 11, fontFamily: "monospace",
                fontWeight: tab === i ? 700 : 400, letterSpacing: "0.1em",
                color: tab === i ? C.amber : C.muted,
                borderBottom: `2px solid ${tab === i ? C.amber : "transparent"}`,
                transition: "all 0.15s",
              }}>{icon} {label}</button>
            ))}
          </div>
        </div>
      </div>

      <div style={{ maxWidth: 860, margin: "0 auto", padding: "24px 20px 0" }}>

        {/* Model status */}
        <div style={{ display: "flex", alignItems: "center", justifySpace: "between", marginBottom: 16 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap", flex: 1 }}>
            <div style={{ width: 6, height: 6, borderRadius: "50%", background: C.teal, boxShadow: `0 0 6px ${C.teal}` }} />
            <span style={{ fontSize: 10, color: C.muted, fontFamily: "monospace", letterSpacing: "0.08em" }}>
              API CONNECTED · Local Fast API · HuggingFace disabled for free offline scaling.
            </span>
          </div>
          <div>
            <input type="file" id="fileDoc" style={{ display: "none" }} onChange={(e) => {
              if (e.target.files && e.target.files[0]) {
                setFile(e.target.files[0]);
                setText("");
              }
            }} accept=".pdf,.docx,.txt" />
            <label htmlFor="fileDoc" style={{
              background: file ? C.border : C.surface, color: file ? C.teal : C.text, border: `1px dashed ${C.borderLight}`, padding: "6px 12px", borderRadius: 6, fontSize: 11, fontFamily: "monospace", cursor: "pointer", transition: "0.2s"
            }}>
              {file ? `📎 ${file.name}` : "📄 UPLOAD DOCUMENT (100+ pages)"}
            </label>
            {file && <span onClick={() => { setFile(null); setText(""); }} style={{ cursor: "pointer", fontSize: 10, marginLeft: 8, color: C.red }}>✕ clear</span>}
          </div>
        </div>

        {/* Live signal preview */}
        {liveSignals && tab === 0 && (
          <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 6, padding: "10px 14px", marginBottom: 14, display: "flex", gap: 20, flexWrap: "wrap" }}>
            <span style={{ fontSize: 10, color: C.muted, fontFamily: "monospace", letterSpacing: "0.08em" }}>LIVE SIGNALS:</span>
            <span style={{ fontSize: 10, fontFamily: "monospace" }}>
              Burstiness CV: <span style={{ color: liveSignals.burstiness.cv < 0.25 ? C.red : C.teal }}>{liveSignals.burstiness.cv}</span>
            </span>
            <span style={{ fontSize: 10, fontFamily: "monospace" }}>
              TTR: <span style={{ color: liveSignals.ttr.ttr < 0.4 ? C.red : C.teal }}>{liveSignals.ttr.ttr}</span>
            </span>
            <span style={{ fontSize: 10, fontFamily: "monospace" }}>
              Transitions: <span style={{ color: liveSignals.transitions.count > 2 ? C.red : C.teal }}>{liveSignals.transitions.count}</span>
            </span>
            <span style={{ fontSize: 10, fontFamily: "monospace" }}>
              Local AI%: <span style={{ color: liveSignals.localAI > 60 ? C.red : C.teal }}>{liveSignals.localAI}%</span>
            </span>
          </div>
        )}

        {/* Input */}
        <div style={{ position: "relative", marginBottom: 16 }}>
          <textarea value={text} onChange={e => setText(e.target.value)} rows={7}
            placeholder={[
              "Paste text to analyze — the detector measures burstiness (CV of sentence lengths), type-token ratio, transition word density, and runs RoBERTa if you've added an HF token...",
              "Paste AI-generated text — the T5 paraphraser will rewrite it to increase perplexity and burstiness to make it sound human...",
              "Paste text to check for internal repetition — uses Jaccard n-gram similarity across all sentence pairs, no API needed..."
            ][tab]}
            style={{
              width: "100%", background: C.surface, border: `1px solid ${C.border}`,
              color: C.text, padding: "14px", borderRadius: 8, fontSize: 13,
              lineHeight: 1.75, resize: "none", transition: "border-color 0.2s"
            }} />
          <div style={{ position: "absolute", bottom: 10, right: 12, fontSize: 10, color: C.muted, fontFamily: "monospace" }}>
            {wordCount}w · {text.length}c
          </div>
        </div>

        {/* Actions */}
        <div style={{ display: "flex", gap: 10, marginBottom: 24, flexWrap: "wrap" }}>
          <button onClick={actions[tab]} disabled={loading || (!text.trim() && !file)}
            style={{
              background: loading ? C.surface : `linear-gradient(135deg, #1a3060, #0d2050)`,
              border: `1px solid ${loading ? C.border : C.borderLight}`,
              color: loading ? C.muted : C.amber,
              padding: "10px 24px", borderRadius: 5, cursor: loading || (!text.trim() && !file) ? "not-allowed" : "pointer",
              fontSize: 11, fontWeight: 700, fontFamily: "monospace", letterSpacing: "0.1em",
              display: "flex", alignItems: "center", gap: 8, opacity: (!text.trim() && !file) ? 0.4 : 1,
              transition: "all 0.2s"
            }}>
            {loading ? <><Spinner /> PROCESSING {progress > 0 && progress < 100 ? `${progress}%` : "..."}</> : actionLabels[tab]}
          </button>
          {loading && (
            <button onClick={abort} style={{ background: "none", border: `1px solid ${C.red}40`, color: C.red, padding: "10px 16px", borderRadius: 5, cursor: "pointer", fontSize: 11, fontFamily: "monospace" }}>✕ STOP</button>
          )}
          <button onClick={() => { setText(""); setFile(null); setDetectResult(null); setHumanizeResult(null); setPlagResult(null); setError(""); document.getElementById("fileDoc").value = ""; }}
            style={{ background: "none", border: `1px solid ${C.border}`, color: C.muted, padding: "10px 16px", borderRadius: 5, cursor: "pointer", fontSize: 11, fontFamily: "monospace" }}>
            CLEAR
          </button>
        </div>

        {/* Error */}
        {error && (
          <div style={{ background: "#180505", border: `1px solid ${C.red}40`, borderRadius: 6, padding: "12px 14px", marginBottom: 20, color: C.red, fontSize: 12, fontFamily: "monospace", lineHeight: 1.7 }}>
            ⚠ {error}
          </div>
        )}

        {/* ══ DETECT RESULTS ══════════════════════════════ */}
        {tab === 0 && detectResult && (
          <div className="panel" style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            <VerdictBanner verdict={detectResult.verdict} score={detectResult.finalAI} mode="ai" />

            {/* Score breakdown */}
            <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 8, padding: 18 }}>
              <div style={{ fontSize: 10, color: C.muted, fontFamily: "monospace", letterSpacing: "0.1em", marginBottom: 14 }}>SCORE BREAKDOWN</div>
              {detectResult.usedModel && (
                <InfoRow label="RoBERTa (fine-tuned AI detector)" value={`${detectResult.robertaScore}% AI`} note="weight: 65%" />
              )}
              <InfoRow label="Local signal composite" value={`${detectResult.local.localAI}% AI`} note={detectResult.usedModel ? "weight: 35%" : "only signal (add HF token for RoBERTa)"} />
              <InfoRow label="Burstiness (CV of sentence lengths)" value={`CV = ${detectResult.local.burstiness.cv}`} note={detectResult.local.burstiness.label} />
              <InfoRow label="Type-Token Ratio" value={`${detectResult.local.ttr.ttr} (${detectResult.local.ttr.unique}u / ${detectResult.local.ttr.total}w)`} note={detectResult.local.ttr.aiScore > 60 ? "low diversity" : "good diversity"} />
              <InfoRow label="AI transition words found" value={`${detectResult.local.transitions.count} (${detectResult.local.transitions.density}%)`} />
              <InfoRow label="Trigram repetition ratio" value={detectResult.local.repetition.ratio} />

              {detectResult.local.transitions.found.length > 0 && (
                <div style={{ marginTop: 10 }}>
                  <div style={{ fontSize: 10, color: C.muted, fontFamily: "monospace", marginBottom: 6, letterSpacing: "0.08em" }}>FLAGGED TRANSITION WORDS:</div>
                  <div>{detectResult.local.transitions.found.map(t => <Pill key={t} text={t} color={C.red} />)}</div>
                </div>
              )}
            </div>

            {/* Signal gauges */}
            <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 8, padding: 18 }}>
              <div style={{ fontSize: 10, color: C.muted, fontFamily: "monospace", letterSpacing: "0.1em", marginBottom: 14 }}>SIGNAL GAUGES (→ right = AI-like)</div>
              <GaugeBar label="BURSTINESS" value={`CV=${detectResult.local.burstiness.cv}`} aiScore={100 - detectResult.local.burstiness.score} helpText="Low CV = uniform sentence lengths = AI signature" />
              <GaugeBar label="VOCAB DIVERSITY (TTR)" value={detectResult.local.ttr.ttr} aiScore={detectResult.local.ttr.aiScore} helpText="Low type-token ratio = repetitive vocabulary = AI" />
              <GaugeBar label="TRANSITION WORD DENSITY" value={`${detectResult.local.transitions.density}%`} aiScore={detectResult.local.transitions.aiScore} helpText="High density = robotic connectors = AI" />
              <GaugeBar label="TRIGRAM REPETITION" value={detectResult.local.repetition.ratio} aiScore={detectResult.local.repetition.aiScore} helpText="Repeated 3-word phrases = AI pattern" />
            </div>

            {/* Burstiness chart */}
            <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 8, padding: 18 }}>
              <div style={{ fontSize: 10, color: C.muted, fontFamily: "monospace", letterSpacing: "0.1em", marginBottom: 2 }}>BURSTINESS VISUALIZATION</div>
              <BurstinessChart lengths={detectResult.local.burstiness.lengths} mean={detectResult.local.burstiness.mean} />
            </div>

            {/* Sentence-level highlight */}
            <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 8, padding: 18 }}>
              <SentenceHighlight sentences={detectResult.local.burstiness.sentences} burstiness={detectResult.local.burstiness} />
            </div>
          </div>
        )}

        {/* ══ HUMANIZER RESULTS ═══════════════════════════ */}
        {tab === 1 && humanizeResult && (
          <div className="panel" style={{ display: "flex", flexDirection: "column", gap: 16 }}>

            {/* Before / After metrics */}
            <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 8, padding: 18 }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 14 }}>
                <div style={{ fontSize: 10, color: C.muted, fontFamily: "monospace", letterSpacing: "0.1em" }}>BEFORE → AFTER COMPARISON</div>
                {humanizeResult.iterations && (
                  <div style={{ fontSize: 10, color: C.teal, fontFamily: "monospace" }}>Took {humanizeResult.iterations} iteration(s)</div>
                )}
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
                {[["ORIGINAL", humanizeResult.before], ["HUMANIZED", humanizeResult.after]].map(([label, sig]) => (
                  <div key={label} style={{ background: C.bg, borderRadius: 6, padding: "12px 14px" }}>
                    <div style={{ fontSize: 10, fontFamily: "monospace", color: C.muted, letterSpacing: "0.08em", marginBottom: 10 }}>{label}</div>
                    <InfoRow label="Burstiness CV" value={sig.burstiness.cv} />
                    <InfoRow label="TTR" value={sig.ttr.ttr} />
                    <InfoRow label="Transitions" value={sig.transitions.count} />
                    <InfoRow label="Est. AI%" value={`${sig.localAI}%`} />
                  </div>
                ))}
              </div>

              {/* Delta */}
              <div style={{ marginTop: 14, background: `${C.teal}10`, border: `1px solid ${C.teal}30`, borderRadius: 5, padding: "10px 14px" }}>
                <div style={{ fontSize: 10, color: C.muted, fontFamily: "monospace", marginBottom: 6 }}>IMPROVEMENT</div>
                <div style={{ display: "flex", gap: 20, flexWrap: "wrap" }}>
                  {[
                    ["Burstiness", humanizeResult.after.burstiness.cv - humanizeResult.before.burstiness.cv, true],
                    ["TTR", humanizeResult.after.ttr.ttr - humanizeResult.before.ttr.ttr, true],
                    ["AI% delta", humanizeResult.after.localAI - humanizeResult.before.localAI, false],
                  ].map(([label, delta, higherGood]) => {
                    const good = higherGood ? delta > 0 : delta < 0;
                    return (
                      <span key={label} style={{ fontSize: 11, fontFamily: "monospace" }}>
                        {label}: <span style={{ color: good ? C.teal : C.red }}>{delta > 0 ? "+" : ""}{delta.toFixed(3)}</span>
                      </span>
                    );
                  })}
                </div>
              </div>
            </div>

            {/* Output */}
            <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 8, padding: 18 }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
                <div style={{ fontSize: 10, color: C.muted, fontFamily: "monospace", letterSpacing: "0.1em" }}>T5 PARAPHRASE OUTPUT</div>
                <button onClick={() => { navigator.clipboard.writeText(humanizeResult.humanized); setCopied(true); setTimeout(() => setCopied(false), 2000); }}
                  style={{ background: copied ? `${C.teal}20` : C.bg, border: `1px solid ${copied ? C.teal : C.border}`, color: copied ? C.teal : C.muted, padding: "5px 12px", borderRadius: 4, cursor: "pointer", fontSize: 10, fontFamily: "monospace" }}>
                  {copied ? "✓ COPIED" : "⎘ COPY"}
                </button>
              </div>
              <div style={{ fontSize: 13, lineHeight: 1.8, color: C.text, background: C.bg, borderRadius: 5, padding: "14px", borderLeft: `3px solid ${C.teal}`, maxHeight: 320, overflowY: "auto" }}>
                {humanizeResult.humanized}
              </div>
            </div>

            {/* After burstiness chart */}
            <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 8, padding: 18 }}>
              <div style={{ fontSize: 10, color: C.muted, fontFamily: "monospace", letterSpacing: "0.1em", marginBottom: 4 }}>BURSTINESS AFTER HUMANIZATION</div>
              <BurstinessChart lengths={humanizeResult.after.burstiness.lengths} mean={humanizeResult.after.burstiness.mean} />
            </div>
          </div>
        )}

        {/* ══ PLAGIARISM RESULTS ══════════════════════════ */}
        {tab === 2 && plagResult && (
          <div className="panel" style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            <VerdictBanner verdict={plagResult.verdict} score={plagResult.originality} mode="originality" />

            {/* Similarity matrix */}
            <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 8, padding: 18, overflowX: "auto" }}>
              <SimilarityHeatmap sentences={plagResult.sentences} pairs={plagResult.pairs} />
            </div>

            {/* Flagged pairs */}
            {plagResult.pairs.length > 0 && (
              <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 8, padding: 18 }}>
                <div style={{ fontSize: 10, color: C.muted, fontFamily: "monospace", letterSpacing: "0.1em", marginBottom: 14 }}>
                  HIGH-SIMILARITY SENTENCE PAIRS ({plagResult.pairs.length})
                </div>
                {plagResult.pairs.map((p, i) => (
                  <div key={i} style={{ marginBottom: 14, padding: "10px 12px", background: C.bg, borderRadius: 5, borderLeft: `3px solid ${p.sim > 0.65 ? C.red : C.amber}` }}>
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
                      <span style={{ fontSize: 10, color: C.muted, fontFamily: "monospace" }}>S{p.i + 1} vs S{p.j + 1}</span>
                      <span style={{ fontSize: 11, fontFamily: "monospace", color: p.sim > 0.65 ? C.red : C.amber }}>{(p.sim * 100).toFixed(0)}% similar</span>
                    </div>
                    <div style={{ fontSize: 12, color: C.text, marginBottom: 4, fontStyle: "italic", opacity: 0.8 }}>"{p.sA}"</div>
                    <div style={{ fontSize: 12, color: C.text, fontStyle: "italic", opacity: 0.8 }}>"{p.sB}"</div>
                  </div>
                ))}
              </div>
            )}

            {/* Stock phrases */}
            {plagResult.foundStock.length > 0 && (
              <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 8, padding: 18 }}>
                <div style={{ fontSize: 10, color: C.muted, fontFamily: "monospace", letterSpacing: "0.1em", marginBottom: 10 }}>
                  STOCK / CLICHÉ PHRASES DETECTED
                </div>
                <div>{plagResult.foundStock.map(p => <Pill key={p} text={p} color={C.amber} />)}</div>
              </div>
            )}

            <div style={{ fontSize: 10, color: C.dim, fontFamily: "monospace", lineHeight: 1.8, padding: "10px 14px", background: C.surface, borderRadius: 6, border: `1px solid ${C.border}` }}>
              METHOD: Jaccard word-set similarity across all sentence pairs. Flags internal repetition and stock phrases.
              For cross-document plagiarism, use <span style={{ color: C.blue }}>Copyleaks</span> or <span style={{ color: C.blue }}>Turnitin</span>.
            </div>
          </div>
        )}

        {/* Empty states */}
        {tab === 0 && !detectResult && !loading && !error && (
          <div style={{ textAlign: "center", padding: "50px 0", color: C.muted }}>
            <div style={{ fontSize: 36, marginBottom: 12, opacity: 0.4 }}>🧬</div>
            <div style={{ fontSize: 11, fontFamily: "monospace", letterSpacing: "0.08em", lineHeight: 2 }}>
              RoBERTa classifier (AI vs Human pairs)<br />
              + burstiness · TTR · transition density<br />
              → combined AI probability score
            </div>
          </div>
        )}
        {tab === 1 && !humanizeResult && !loading && !error && (
          <div style={{ textAlign: "center", padding: "50px 0", color: C.muted }}>
            <div style={{ fontSize: 36, marginBottom: 12, opacity: 0.4 }}>↺</div>
            <div style={{ fontSize: 11, fontFamily: "monospace", letterSpacing: "0.08em", lineHeight: 2 }}>
              T5 paraphraser trained on paraphrase pairs<br />
              increases perplexity + burstiness<br />
              requires HF token · ⚙ Config
            </div>
          </div>
        )}
        {tab === 2 && !plagResult && !loading && !error && (
          <div style={{ textAlign: "center", padding: "50px 0", color: C.muted }}>
            <div style={{ fontSize: 36, marginBottom: 12, opacity: 0.4 }}>⊕</div>
            <div style={{ fontSize: 11, fontFamily: "monospace", letterSpacing: "0.08em", lineHeight: 2 }}>
              Jaccard n-gram similarity matrix<br />
              across all sentence pairs<br />
              fully client-side · no API needed
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
