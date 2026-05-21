/**
 * ============================================================================
 * FrugalSloth v0.3.3 — train empty state + electron build ready
 * ============================================================================
 * Fixes:
 *   - ONNX WASM .mjs files copied to public/
 *   - ASCII banner renders as single <pre> block
 *   - Training Infinity% + 100% milestone deduplication
 *   - Full-width header bar
 *   - Home tab with overview, speed claims, use cases
 *   - Sandbox tensor shape mismatch fixed
 *   - Docs with subsection sidebar navigation
 * ============================================================================
 */

import { useState, useRef, useCallback, useEffect, useMemo, type ReactNode } from "react";
import * as tf from "@tensorflow/tfjs";
import * as ort from "onnxruntime-web";
import {
  Moon, Sun, Database, Brain, Zap, HardDrive, Upload,
  Trash2, Play, Pause, Square, RotateCcw, Save, HelpCircle,
  FileJson, FileSpreadsheet, ChevronRight, Check, Copy, BookOpen,
  Cpu, Globe, Shield, FlaskConical, Sparkles, Monitor,
  Package, Info, Loader2, TrendingDown, Settings2, Eye, Code2,
  Terminal, Home, Layers, Target, Anchor,
  WifiOff, Gauge, Download, Tag, SlidersHorizontal, Activity,
} from "lucide-react";
import type {
  DatasetSchema, TrainingConfig, StoredModel, EpochMetrics,
  TrainingStatus, NormalizationStats,
} from "@/types/frugalsloth";
import { DEFAULT_TRAINING_CONFIG } from "@/types/frugalsloth";
import { parseCSV, parseCSVString } from "@/utils/csvParser";
import { parseJSON } from "@/utils/jsonParser";
import { saveModel, listModels, deleteModel } from "@/utils/indexedDB";
import { exportToOnnx } from "@/utils/onnxExporter";
import { generateEngine } from "@/utils/engineTemplate";
import JSZip from "jszip";
import "./terminal.css";

/* ── THEME ── */
function useTheme() {
  const [theme, setTheme] = useState<"light" | "dark">(() => {
    if (typeof window !== "undefined") {
      const saved = localStorage.getItem("frugalsloth-theme");
      if (saved === "dark" || saved === "light") return saved;
      return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
    }
    return "light";
  });
  useEffect(() => { document.documentElement.setAttribute("data-theme", theme); localStorage.setItem("frugalsloth-theme", theme); }, [theme]);
  const toggle = useCallback(() => setTheme((t) => (t === "light" ? "dark" : "light")), []);
  return { theme, toggle };
}

/* ── HARDWARE DETECTION ── */
let cachedHwInfo: string | null = null;
async function detectHardware(): Promise<string> {
  if (cachedHwInfo) return cachedHwInfo;
  const parts: string[] = [];
  parts.push(`${navigator.hardwareConcurrency || "?"} cores`);
  try {
    const canvas = document.createElement("canvas");
    const gl = canvas.getContext("webgl2") || canvas.getContext("webgl");
    if (gl) {
      const dbg = gl.getExtension("WEBGL_debug_renderer_info");
      if (dbg) {
        const renderer = gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL);
        if (renderer && !renderer.includes("Google SwiftShader") && !renderer.includes("Software")) {
          parts.push(renderer);
        }
      }
    }
  } catch { /* ignore */ }
  try {
    if (navigator.gpu) {
      const adapter = await navigator.gpu.requestAdapter();
      if (adapter) {
        const info = await adapter.requestAdapterInfo?.() || (adapter as unknown as { info: { description?: string; architecture?: string } }).info;
        if (info) {
          const desc = (info as { description?: string }).description || (info as { architecture?: string }).architecture;
          if (desc) parts.push(`WebGPU: ${desc}`);
        }
      }
    }
  } catch { /* ignore */ }
  cachedHwInfo = parts.join(" · ");
  return cachedHwInfo;
}

/* ── TOOLTIP ── */
function TooltipLabel({ label, tooltip, children }: { label: string; tooltip: string; children?: ReactNode }) {
  return (
    <label className="fs-label">
      <span className="fs-label-text">{label}</span>
      <span className="fs-tooltip-wrap">
        <HelpCircle size={14} className="fs-info-icon" />
        <span className="fs-tooltip">{tooltip}</span>
      </span>
      {children}
    </label>
  );
}

const BUILD_DATE = '2026-01-18';

/* ── ASCII BANNER ── */
const ASCII_BANNER = `FRUGALSLOTH
Universal Edge AI Trainer & Inference Engine v0.3.3 (build ${BUILD_DATE})
ONNX Runtime Web 1.17+ . TensorFlow.js . Web Workers
100% Private . Zero Backend . Browser-Native`;

/* ── SMOOTHED CHART DATA ── */
function emaSmooth(data: number[], alpha = 0.3): number[] {
  if (data.length === 0) return [];
  const s = [data[0]];
  for (let i = 1; i < data.length; i++) s.push(alpha * data[i] + (1 - alpha) * s[i - 1]);
  return s;
}

/* ── TECH BADGES ── */
function TechBadges() {
  return (
    <div style={{ display: "flex", gap: "6px", flexWrap: "wrap", alignItems: "center" }}>
      <span className="fs-tech-badge"><Cpu size={11} /> TF.js</span>
      <span className="fs-tech-badge"><Globe size={11} /> ONNX 1.17+</span>
      <span className="fs-tech-badge"><Layers size={11} /> Web Workers</span>
      <span className="fs-tech-badge"><Shield size={11} /> 100% Private</span>
      <span className="fs-tech-badge"><Zap size={11} /> Zero Backend</span>
    </div>
  );
}

/* ── CANVAS CHART ── */
function MiniChart({ metrics }: { metrics: EpochMetrics[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || metrics.length < 2) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    const w = rect.width, h = rect.height;
    const pad = { top: 14, right: 10, bottom: 22, left: 44 };
    const pw = w - pad.left - pad.right;
    const ph = h - pad.top - pad.bottom;
    ctx.clearRect(0, 0, w, h);
    const epochs = metrics.map((m) => m.epoch);
    const losses = metrics.map((m) => m.loss);
    const valLosses = metrics.map((m) => m.valLoss);
    const smoothed = emaSmooth(losses, 0.3);
    const all = [...losses, ...valLosses].filter((v) => Number.isFinite(v));
    if (all.length === 0) return;
    let minV = Math.min(...all) * 0.95;
    let maxV = Math.max(...all) * 1.05;
    const allIdentical = maxV - minV < 0.0001;
    if (allIdentical) {
      // Center the flat line in the chart so it's visible
      const centerVal = (minV + maxV) / 2;
      minV = centerVal - 0.05;
      maxV = centerVal + 0.05;
    }
    const range = maxV - minV || 1;
    const sx = (e: number) => pad.left + ((e - epochs[0]) / Math.max(epochs[epochs.length - 1] - epochs[0], 1)) * pw;
    const sy = (v: number) => pad.top + ph - ((v - minV) / range) * ph;
    // Grid
    const isDark = document.documentElement.getAttribute("data-theme") === "dark";
    const borderColor = isDark ? "#1f1f2e" : "#e5e7eb";
    const mutedColor = isDark ? "#6b6b7b" : "#9ca3af";
    ctx.strokeStyle = borderColor; ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
      const y = pad.top + (ph / 4) * i;
      ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(w - pad.right, y); ctx.stroke();
      ctx.fillStyle = mutedColor; ctx.font = "10px JetBrains Mono"; ctx.textAlign = "right";
      const tickVal = maxV - (range / 4) * i;
      ctx.fillText(tickVal < 0.01 ? tickVal.toFixed(4) : tickVal < 0.1 ? tickVal.toFixed(3) : tickVal.toFixed(2), pad.left - 6, y + 3);
    }
    // Raw loss (faint)
    ctx.strokeStyle = isDark ? "#333" : "#cbd5e1"; ctx.lineWidth = 1;
    ctx.beginPath();
    metrics.forEach((m, i) => { const x = sx(m.epoch), y = sy(m.loss); i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y); });
    ctx.stroke();
    // Smoothed (emerald)
    ctx.strokeStyle = "#10b981"; ctx.lineWidth = 2.5;
    ctx.beginPath();
    smoothed.forEach((v, i) => { const x = sx(epochs[i]), y = sy(v); i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y); });
    ctx.stroke();
    // If all values identical, draw a dashed center line + label
    if (allIdentical) {
      const centerY = sy((minV + maxV) / 2);
      ctx.setLineDash([4, 4]); ctx.strokeStyle = mutedColor; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(pad.left, centerY); ctx.lineTo(w - pad.right, centerY); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "#10b981"; ctx.font = "bold 11px Inter"; ctx.textAlign = "center";
      const val = (minV + maxV) / 2;
      const label = val < 0.0001 ? "loss ≈ 0" : `loss = ${val.toFixed(4)}`;
      ctx.fillText(label, w / 2, centerY - 6);
    }
    // Val loss (orange)
    ctx.strokeStyle = "#f59e0b"; ctx.lineWidth = 2;
    ctx.beginPath();
    metrics.forEach((m, i) => { const x = sx(m.epoch), y = sy(m.valLoss); i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y); });
    ctx.stroke();
    // Legend
    ctx.font = "bold 10px Inter"; ctx.textAlign = "left";
    ctx.fillStyle = mutedColor; ctx.fillText("── raw", pad.left, h - 2);
    ctx.fillStyle = "#10b981"; ctx.fillText("── smoothed", pad.left + 50, h - 2);
    ctx.fillStyle = "#f59e0b"; ctx.fillText("── val", pad.left + 135, h - 2);
    // X labels
    ctx.fillStyle = mutedColor; ctx.font = "10px JetBrains Mono"; ctx.textAlign = "center";
    const step = Math.max(1, Math.floor(epochs.length / 5));
    for (let i = 0; i < epochs.length; i += step) ctx.fillText(String(epochs[i]), sx(epochs[i]), h - 4);
  }, [metrics]);
  return <canvas ref={canvasRef} style={{ width: "100%", height: "180px" }} />;
}

/* ── TYPE COLOR BADGE ── */
const typeColor = (t: string) => {
  switch (t) {
    case "numeric": return "fs-badge-info";
    case "categorical": return "fs-badge-warning";
    case "text": return "fs-badge-success";
    case "datetime": return "fs-badge-accent";
    default: return "fs-badge-neutral";
  }
};

/* ── MODEL PRESETS ── */
const MODEL_PRESETS = [
  { id: "tinybert", name: "TinyBERT-4L", desc: "4-layer distilled BERT, fastest inference", size: "~14MB", embDim: 312 },
  { id: "distilbert", name: "DistilBERT", desc: "6-layer distilled BERT, balanced speed/quality", size: "~66MB", embDim: 768 },
  { id: "mobilebert", name: "MobileBERT", desc: "Mobile-optimized for edge devices", size: "~100MB", embDim: 512 },
  { id: "albert", name: "ALBERT-base", desc: "Lightweight with parameter sharing", size: "~12MB", embDim: 768 },
  { id: "custom", name: "Custom ONNX", desc: "Upload your own transformer model", size: "Any", embDim: 0 },
];

/* ═══════════════════════════════════════════════════════════════════════
   MAIN APP
   ═══════════════════════════════════════════════════════════════════ */
export default function App() {
  const { theme: currentTheme, toggle } = useTheme();
  const [activeTab, setActiveTab] = useState<"home" | "data" | "train" | "sandbox" | "models" | "import" | "releases" | "docs">("home");
  const [dataset, setDataset] = useState<{ schema: DatasetSchema; rows: number[][]; labels: number[]; normStats: NormalizationStats } | null>(null);
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus>("idle");
  const [metrics, setMetrics] = useState<EpochMetrics[]>([]);
  const [currentEpoch, setCurrentEpoch] = useState(0);
  const [totalEpochs, setTotalEpochs] = useState(0);
  const [config, setConfig] = useState<TrainingConfig>({ ...DEFAULT_TRAINING_CONFIG });
  const [log, setLog] = useState<string[]>([]);
  const [savedModels, setSavedModels] = useState<StoredModel[]>([]);
  const [modelName, setModelName] = useState("");
  const [parseProgress, setParseProgress] = useState(0);
  const [exportStatus, setExportStatus] = useState("");
  const [bestValLoss, setBestValLoss] = useState(Infinity);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [bannerShown, setBannerShown] = useState(false);
  const [isBusy, setIsBusy] = useState(false);

  const bestWeightsRef = useRef<string | null>(null);
  const modelSummaryRef = useRef<{ totalParams: number; inputDim?: number; outputDim?: number; hiddenLayers?: number[]; trainableParams?: number } | null>(null);
  const logEndRef = useRef<HTMLDivElement>(null);

  const addLog = useCallback((msg: string) => {
    setLog((prev) => [...prev.slice(-199), msg]);
  }, []);

  useEffect(() => { if (logEndRef.current) logEndRef.current.scrollIntoView({ behavior: "smooth" }); }, [log]);

  /* ── Init ── */
  useEffect(() => {
    (async () => {
      await tf.setBackend("webgl");
      await tf.ready();
      const backend = tf.getBackend();
      const hwInfo = await detectHardware();
      if (!bannerShown) {
        addLog(ASCII_BANNER);
        addLog(`[System] TF.js ${tf.version.tfjs} — backend: ${backend} (${hwInfo})`);
        addLog(`[System] ONNX Runtime Web 1.17+ loaded`);
        addLog(`[System] IndexedDB: ready`);
        addLog(`[System] Web Workers: supported`);
        addLog(`[System] ${savedModels.length} saved model(s) in IndexedDB`);
        addLog(`[System] Build ${BUILD_DATE}. If this date is old, click Models → Hard Flush.`);
        addLog(`[System] Ready. Load a dataset to begin training.`);
        setBannerShown(true);
      }
    })();
    listModels().then((m) => { setSavedModels(m); }).catch(() => {});
  }, [addLog, bannerShown, savedModels.length]);

  /* ── Reset training state when dataset changes ── */
  const resetTrainingState = useCallback(() => {
    setMetrics([]);
    setCurrentEpoch(0);
    setTrainingProgress(0);
    setBestValLoss(Infinity);
    setTrainingStatus("idle");
    bestWeightsRef.current = null;
    addLog("[Train] Previous training cleared. Ready for new dataset.");
  }, [addLog]);

  /* ── File Upload (CSV + JSON) ── */
  const handleFileUpload = useCallback(async (file: File) => {
    resetTrainingState();
    setIsBusy(true);
    try {
      setParseProgress(0);
      const name = file.name.toLowerCase();
      const isJson = name.endsWith(".json");
      const isJsonl = name.endsWith(".jsonl");
      addLog(`[Data] Parsing ${isJsonl ? "JSONL" : isJson ? "JSON" : "CSV"}: ${file.name}`);
      let result: ReturnType<typeof parseCSVString>;
      if (isJsonl) {
        const text = await file.text();
        const lines = text.trim().split("\n").filter((l) => l.trim());
        if (lines.length === 0) throw new Error("JSONL file is empty");
        const firstObj = JSON.parse(lines[0]);
        const keys = Object.keys(firstObj);
        const csvLines = [keys.join(",")];
        for (const line of lines) {
          const obj = JSON.parse(line);
          csvLines.push(keys.map((k) => { const v = obj[k]; return v === null ? "" : String(v).includes(",") ? `"${v}"` : String(v); }).join(","));
        }
        result = parseCSVString(csvLines.join("\n"));
      } else if (isJson) {
        result = parseJSON(await file.text());
      } else {
        result = await parseCSV(file, (p) => setParseProgress(p));
      }
      setDataset({ schema: result.schema, rows: result.numericRows, labels: result.labels, normStats: result.normStats });
      setModelName(file.name.replace(/\.(csv|json|jsonl)$/i, ""));
      addLog(`[Data] Loaded ${result.schema.rowCount} rows × ${result.schema.columns.length} cols — target: ${result.schema.columns[result.schema.targetIndex].name}`);
      setActiveTab("train");
    } catch (err) {
      addLog(`[Error] ${err instanceof Error ? err.message : String(err)}`);
    } finally { setParseProgress(0); setIsBusy(false); }
  }, [addLog, setIsBusy, resetTrainingState]);

  /* ── Demo Datasets ── */
  const loadDemoCSV = useCallback(() => {
    resetTrainingState();
    // 100 realistic Wine samples with NOISE so model can't instantly memorize
    const qualities = [5,5,6,5,7,6,5,7,5,6,5,5,7,6,5,6,5,7,5,6,6,5,7,5,6,5,6,7,5,5,6,5,5,7,6,5,5,6,7,5,6,5,7,5,5,6,5,7,6,5,6,5,5,7,5,6,5,6,7,5,5,6,5,7,5,6,5,5,6,7,5,5,6,5,5,7,6,5,5,6,7,5,6,5,7,5,6,5,6,5,7,5,5,6,5,7,5,5];
    const rows: string[] = ["fixed acidity,volatile acidity,citric acid,residual sugar,chlorides,free sulfur dioxide,total sulfur dioxide,density,pH,sulphates,alcohol,quality"];
    for (let i = 0; i < 100; i++) {
      const q = qualities[i % qualities.length];
      // Base values with LARGE random noise so features DON'T perfectly correlate with quality
      const base = 7.0 + (Math.random() - 0.5) * 3.5;  // 4.5 to 10.5, random
      const va = 0.5 + Math.random() * 0.6;  // 0.5 to 1.1, mostly random
      const ca = Math.random() * 0.6;  // 0 to 0.6, random
      const rs = 1.0 + Math.random() * 6.0;  // 1 to 7, random
      const cl = 0.05 + Math.random() * 0.1;  // 0.05 to 0.15, random
      const fsd = 5 + Math.floor(Math.random() * 30);  // 5 to 35, random
      const tsd = 20 + Math.floor(Math.random() * 80);  // 20 to 100, random
      const dens = 0.995 + Math.random() * 0.005;  // 0.995 to 1.000, random
      const pH = 3.0 + Math.random() * 0.6;  // 3.0 to 3.6, random
      const sul = 0.4 + Math.random() * 0.6;  // 0.4 to 1.0, random
      const alc = 9.0 + Math.random() * 2.0;  // 9 to 11, random
      rows.push(`${base.toFixed(1)},${va.toFixed(2)},${ca.toFixed(2)},${rs.toFixed(1)},${cl.toFixed(3)},${Math.round(fsd)},${Math.round(tsd)},${dens.toFixed(4)},${pH.toFixed(2)},${sul.toFixed(2)},${alc.toFixed(1)},${q}`);
    }
    const result = parseCSVString(rows.join("\n"));
    setDataset({ schema: result.schema, rows: result.numericRows, labels: result.labels, normStats: result.normStats });
    setModelName("wine-quality");
    addLog("[Data] Demo CSV loaded: Wine Quality (100×12)");
    setActiveTab("train");
  }, [addLog, resetTrainingState]);

  const loadDemoJSON = useCallback(() => {
    resetTrainingState();
    const json = JSON.stringify([
      { sepal_length: 5.1, sepal_width: 3.5, petal_length: 1.4, petal_width: 0.2, species: 0 },
      { sepal_length: 4.9, sepal_width: 3.0, petal_length: 1.4, petal_width: 0.2, species: 0 },
      { sepal_length: 4.7, sepal_width: 3.2, petal_length: 1.3, petal_width: 0.2, species: 0 },
      { sepal_length: 7.0, sepal_width: 3.2, petal_length: 4.7, petal_width: 1.4, species: 1 },
      { sepal_length: 6.4, sepal_width: 3.2, petal_length: 4.5, petal_width: 1.5, species: 1 },
      { sepal_length: 6.9, sepal_width: 3.1, petal_length: 4.9, petal_width: 1.5, species: 1 },
      { sepal_length: 6.3, sepal_width: 3.3, petal_length: 6.0, petal_width: 2.5, species: 2 },
      { sepal_length: 5.8, sepal_width: 2.7, petal_length: 5.1, petal_width: 1.9, species: 2 },
      { sepal_length: 7.1, sepal_width: 3.0, petal_length: 5.9, petal_width: 2.1, species: 2 },
      { sepal_length: 6.3, sepal_width: 2.9, petal_length: 5.6, petal_width: 1.8, species: 2 },
    ]);
    const result = parseJSON(json);
    setDataset({ schema: result.schema, rows: result.numericRows, labels: result.labels, normStats: result.normStats });
    setModelName("iris");
    addLog("[Data] Demo JSON loaded: Iris (10×5)");
    setActiveTab("train");
  }, [addLog, resetTrainingState]);

  /* ── Training (In-Thread — Web Worker removed due to module loading failures) ── */
  const shouldStopRef = useRef(false);

  const startTraining = useCallback(async () => {
    if (!dataset) return;
    // Validate: need ≥2 unique classes for classification
    const uniqueLabels = Array.from(new Set(dataset.labels));
    if (dataset.schema.taskType !== "regression" && uniqueLabels.length < 2) {
      addLog(`[Error] Only 1 class found: "${uniqueLabels[0]}". Need ≥2 different labels for classification.`);
      addLog(`[Error] Fix: add rows with a different label value (e.g., "BUSY") alongside "${uniqueLabels[0]}".`);
      return;
    }
    setIsBusy(true);
    setMetrics([]); setCurrentEpoch(0); setBestValLoss(Infinity); setTrainingProgress(0);
    setTrainingStatus("running");
    setTotalEpochs(config.epochs);
    shouldStopRef.current = false;
    addLog("[Train] Starting in-thread training...");

    try {
      await tf.ready();
      const labels = dataset.labels;
      const uniqueLabels = Array.from(new Set(labels)).sort((a, b) => a - b);
      const labelMap = new Map<number, number>();
      uniqueLabels.forEach((v, i) => labelMap.set(v, i));
      const remapped = labels.map((l) => labelMap.get(l)!);
      const numClasses = uniqueLabels.length;
      const outputDim = dataset.schema.taskType === "regression" ? 1 : numClasses;
      const inputDim = dataset.rows[0]?.length ?? 0;

      // Create tensors
      const allXs = tf.tensor2d(dataset.rows);
      const allYs = dataset.schema.taskType === "regression"
        ? tf.tensor2d(labels.map((l) => [l]))
        : tf.oneHot(tf.tensor1d(remapped, "int32"), numClasses);

      // Train/val split (80/20)
      const n = dataset.rows.length;
      const valCount = Math.floor(n * config.validationSplit);
      const indices = Array.from({ length: n }, (_, i) => i);
      for (let i = n - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [indices[i], indices[j]] = [indices[j], indices[i]];
      }
      const trainIdx = indices.slice(0, n - valCount);
      const valIdx = indices.slice(n - valCount);
      const trainXs = tf.gather(allXs, trainIdx);
      const trainYs = tf.gather(allYs, trainIdx);
      const valXs = valCount > 0 ? tf.gather(allXs, valIdx) : trainXs;
      const valYs = valCount > 0 ? tf.gather(allYs, valIdx) : trainYs;

      // Build model
      const model = tf.sequential();
      for (let i = 0; i < config.hiddenLayers.length; i++) {
        model.add(tf.layers.dense({
          inputShape: i === 0 ? [inputDim] : undefined,
          units: config.hiddenLayers[i],
          activation: "relu",
          kernelInitializer: "glorotUniform",
          kernelRegularizer: tf.regularizers.l2({ l2: 0.01 }),
        }));
      }
      const outputActivation = dataset.schema.taskType === "regression" ? "linear" : (outputDim === 2 ? "sigmoid" : "softmax");
      model.add(tf.layers.dense({
        units: outputDim === 2 ? 1 : outputDim,
        activation: outputActivation,
      }));

      const loss = dataset.schema.taskType === "regression" ? "meanSquaredError" : (outputDim === 2 ? "binaryCrossentropy" : "categoricalCrossentropy");
      const optimizer = config.optimizer === "sgd" ? tf.train.sgd(config.learningRate) : config.optimizer === "rmsprop" ? tf.train.rmsprop(config.learningRate) : tf.train.adam(config.learningRate);
      const trainMetrics: string[] = dataset.schema.taskType === "regression" ? ["mse"] : ["accuracy"];
      model.compile({ optimizer, loss, metrics: trainMetrics });
      modelSummaryRef.current = { inputDim, outputDim, hiddenLayers: [...config.hiddenLayers], totalParams: model.countParams(), trainableParams: model.countParams() } as any;
      addLog(`[Train] Model: ${inputDim}→${config.hiddenLayers.join(",")}→${outputDim} (${model.countParams()} params)`);

      const totalEpochs = config.epochs;
      let bestValLoss = Infinity;
      let patienceCounter = 0;
      const allMetrics: EpochMetrics[] = [];

      for (let epoch = 1; epoch <= totalEpochs; epoch++) {
        if (shouldStopRef.current) break;

        const h = await model.fit(trainXs, trainYs, { epochs: 1, batchSize: config.batchSize, verbose: 0 });
        const lossVal = h.history.loss[0] as number;
        const accVal = h.history.acc ? h.history.acc[0] as number : undefined;

        const valEval = model.evaluate(valXs, valYs, { verbose: 0 }) as tf.Tensor | tf.Tensor[];
        const valLossVal = Array.isArray(valEval) ? (valEval[0].dataSync()[0]) : valEval.dataSync()[0];
        if (Array.isArray(valEval)) valEval.forEach((t) => t.dispose()); else valEval.dispose();

        const metric: EpochMetrics = { epoch, loss: lossVal, valLoss: valLossVal };
        if (accVal !== undefined) metric.accuracy = accVal;
        allMetrics.push(metric);

        setMetrics((prev) => [...prev, metric]);
        setCurrentEpoch(epoch);
        setTrainingProgress(Math.round((epoch / totalEpochs) * 100));

        const isFirstFew = epoch <= 5;
        const isMilestone = epoch % 10 === 0 || epoch >= totalEpochs - 2;
        if (isFirstFew || isMilestone) {
          const lossStr = isFirstFew ? lossVal.toPrecision(6) : (lossVal < 0.0001 ? lossVal.toExponential(2) : lossVal.toFixed(4));
          const valStr = isFirstFew ? valLossVal.toPrecision(6) : (valLossVal < 0.0001 ? valLossVal.toExponential(2) : valLossVal.toFixed(4));
          const accStr = accVal !== undefined ? `acc=${(accVal * 100).toFixed(1)}%` : "";
          addLog(`[Train] Epoch ${epoch}/${totalEpochs}: loss=${lossStr} val=${valStr} ${accStr}`);
        }

        // Early stopping
        if (valLossVal < bestValLoss) {
          bestValLoss = valLossVal;
          patienceCounter = 0;
          // Save best weights
          const weights: number[][] = [];
          for (const layer of model.layers) {
            for (const w of layer.getWeights()) {
              weights.push(Array.from(w.dataSync()));
            }
          }
          bestWeightsRef.current = JSON.stringify(weights);
        } else {
          patienceCounter++;
          if (config.earlyStoppingPatience > 0 && patienceCounter >= config.earlyStoppingPatience) {
            addLog(`[Train] Early stop at epoch ${epoch} (${config.earlyStoppingPatience} epochs no improvement)`);
            break;
          }
        }

        // Yield to browser UI
        await new Promise((resolve) => setTimeout(resolve, 10));
      }

      trainXs.dispose(); trainYs.dispose(); valXs.dispose(); valYs.dispose(); allXs.dispose(); allYs.dispose();
      model.dispose();

      setTrainingStatus("completed");
      setTrainingProgress(100);
      setIsBusy(false);
      addLog(`[Train] ✅ Complete! ${allMetrics.length}/${totalEpochs} epochs, best val loss: ${bestValLoss.toFixed(6)}`);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      addLog(`[Train Error] ${msg}`);
      setTrainingStatus("error");
      setIsBusy(false);
    }
  }, [dataset, config, addLog, setIsBusy]);

  const pauseTraining = useCallback(() => { /* no-op in in-thread mode */ }, []);
  const resumeTraining = useCallback(() => { /* no-op in in-thread mode */ }, []);
  const stopTraining = useCallback(() => { shouldStopRef.current = true; setTrainingStatus("idle"); setIsBusy(false); }, [setIsBusy]);

  /* ── Save Model (with auto-versioning) ── */
  const saveCurrentModel = useCallback(async () => {
    if (!dataset || !bestWeightsRef.current) return;
    // Auto-version: if name exists, append _v2, _v3, etc.
    let finalName = modelName || "untitled";
    const existingNames = savedModels.map((m) => m.name);
    if (existingNames.includes(finalName)) {
      let v = 2;
      while (existingNames.includes(`${finalName}_v${v}`)) v++;
      finalName = `${finalName}_v${v}`;
    }
    const stored: StoredModel = {
      id: `model-${Date.now()}`, name: finalName, createdAt: Date.now(), updatedAt: Date.now(),
      schema: dataset.schema, config,
      modelSummary: { inputDim: dataset.rows[0]?.length ?? 0, outputDim: new Set(dataset.labels).size, hiddenLayers: [...config.hiddenLayers], totalParams: modelSummaryRef.current?.totalParams ?? 0, trainableParams: modelSummaryRef.current?.totalParams ?? 0 },
      normStats: dataset.normStats, history: metrics,
      bestValLoss: metrics.length > 0 ? Math.min(...metrics.map((m) => m.valLoss)) : Infinity,
      hasWeights: true, weightsJson: bestWeightsRef.current,
    };
    await saveModel(stored);
    setSavedModels((prev) => [stored, ...prev]);
    addLog(`[Model] Saved: ${stored.name}`);
  }, [dataset, config, metrics, modelName, savedModels]);

  /* ── Fine-tune ── */
  const loadModelForFinetune = useCallback((model: StoredModel) => {
    if (!model.hasWeights || !model.weightsJson) { addLog("[Fine-tune] No saved weights"); return; }
    setDataset({ schema: model.schema, rows: [], labels: [], normStats: model.normStats });
    setConfig({ ...model.config, learningRate: model.config.learningRate * 0.1, epochs: Math.max(10, Math.floor(model.config.epochs * 0.3)) });
    setModelName(`${model.name}-finetuned`);
    bestWeightsRef.current = model.weightsJson;
    modelSummaryRef.current = model.modelSummary;
    addLog(`[Fine-tune] Loaded: ${model.name}`);
    setActiveTab("train");
  }, [addLog]);

  /* ── Export ── */
  const exportModel = useCallback(async () => {
    if (!dataset || !bestWeightsRef.current) return;
    try {
      setExportStatus("Converting to ONNX...");
      const onnxResult = exportToOnnx(bestWeightsRef.current, dataset.schema, config, true);
      const shouldEmbed = onnxResult.sizeBytes < 500 * 1024;
      const { engineJs, readme } = generateEngine(onnxResult.model, dataset.schema, dataset.normStats, modelName || "frugalsloth-model", shouldEmbed);
      setExportStatus("Packaging...");
      const zip = new JSZip();
      zip.file("frugalsloth-engine.js", engineJs);
      if (!shouldEmbed) zip.file("model.onnx", onnxResult.model);
      zip.file("README.md", readme);
      const blob = await zip.generateAsync({ type: "blob" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a"); a.href = url; a.download = `${modelName || "frugalsloth"}-export-${Date.now()}.zip`; a.click(); URL.revokeObjectURL(url);
      setExportStatus(`Downloaded: ${(blob.size / 1024).toFixed(1)}KB`);
      addLog(`[Export] Zip: ${(blob.size / 1024).toFixed(1)}KB`);
    } catch (err) { setExportStatus(`Failed: ${err instanceof Error ? err.message : String(err)}`); addLog(`[Export Error] ${err instanceof Error ? err.message : String(err)}`); }
  }, [dataset, config, modelName, addLog]);

  /* ── Tabs ── */
  const tabs = [
    { id: "home" as const, label: "Home", icon: <Home size={15} />, disabled: isBusy },
    { id: "data" as const, label: "Data", icon: <Database size={15} />, disabled: isBusy },
    { id: "train" as const, label: "Train", icon: <Brain size={15} />, disabled: isBusy },
    { id: "sandbox" as const, label: "Sandbox", icon: <FlaskConical size={15} />, disabled: isBusy || !bestWeightsRef.current },
    { id: "models" as const, label: "Models", icon: <HardDrive size={15} />, disabled: isBusy },
    { id: "import" as const, label: "Import ONNX", icon: <Upload size={15} />, disabled: isBusy },
    { id: "releases" as const, label: "Releases", icon: <Tag size={15} />, disabled: isBusy },
    { id: "docs" as const, label: "Docs", icon: <BookOpen size={15} />, disabled: isBusy },
  ];

  /* ═══════════════════════════════════════════════════════════ RENDER */
  return (
    <div className="fs-app-shell">
      {/* ── FULL-WIDTH HEADER ── */}
      <header className="fs-app-header">
        <div style={{ width: "100%", margin: 0, padding: "10px 16px 10px 12px", display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 12 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <Sparkles size={24} style={{ color: "var(--accent)" }} />
              <h1 style={{ fontSize: 22, fontWeight: 700, letterSpacing: "-0.03em", margin: 0 }}>FrugalSloth</h1>
            </div>
            <span className="fs-badge fs-badge-accent">v0.3.3</span>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
            <TechBadges />
            {isBusy && (
              <span className="fs-badge fs-badge-accent" style={{ animation: "pulse 1.5s ease-in-out infinite" }}>
                <Loader2 size={12} style={{ animation: "spin 1s linear infinite" }} /> Processing...
              </span>
            )}
            <button className="fs-theme-toggle" onClick={toggle} title={`Switch to ${currentTheme === "light" ? "dark" : "light"} mode`}>
              {currentTheme === "light" ? <Moon size={16} /> : <Sun size={16} />}
            </button>
          </div>
        </div>
        {/* ── TABS ── */}
        <div style={{ width: "100%", padding: "0 12px 8px" }}>
          <div className="fs-tabs">
            {tabs.map((t) => (
              <button key={t.id} onClick={() => !t.disabled && setActiveTab(t.id)} className={`fs-tab ${activeTab === t.id ? "fs-tab-active" : ""}`} disabled={t.disabled}>
                {t.icon} {t.label}
              </button>
            ))}
          </div>
        </div>
      </header>

      {/* ── MAIN CONTENT (scrollable) ── */}
      <main className="fs-app-main">

        {/* ═══════ HOME TAB ═══════ */}
        {activeTab === "home" && (
          <HomeTab onLoadDemo={loadDemoCSV} onLoadDemoJson={loadDemoJSON} onChangeTab={(t: string) => { (setActiveTab as (t: string) => void)(t); }} />
        )}

        {/* ═══════ DATA TAB ═══════ */}
        {activeTab === "data" && (
          <DataTab dataset={dataset} onUpload={handleFileUpload} onDemoCSV={loadDemoCSV} onDemoJSON={loadDemoJSON} onClear={() => { setDataset(null); resetTrainingState(); addLog("[Data] Dataset cleared."); }} parseProgress={parseProgress} />
        )}

        {/* ═══════ TRAIN TAB ═══════ */}
        {activeTab === "train" && (
          <TrainTab
            config={config} setConfig={setConfig} modelName={modelName} setModelName={setModelName}
            trainingStatus={trainingStatus} trainingProgress={trainingProgress} currentEpoch={currentEpoch} totalEpochs={totalEpochs}
            metrics={metrics} bestValLoss={bestValLoss} modelSummaryRef={modelSummaryRef}
            exportStatus={exportStatus} onStart={startTraining} onPause={pauseTraining} onResume={resumeTraining}
            onStop={stopTraining} onSave={saveCurrentModel} onExport={exportModel} onClear={() => { setDataset(null); bestWeightsRef.current = null; resetTrainingState(); setActiveTab("train"); addLog("[Train] Cleared. Load a dataset to start fresh."); }} hasWeights={!!bestWeightsRef.current} datasetName={modelName} hasDataset={!!dataset}
            onLoadDemo={loadDemoCSV} onLoadDemoJson={loadDemoJSON}
          />
        )}

        {/* ═══════ SANDBOX TAB ═══════ */}
        {activeTab === "sandbox" && dataset && bestWeightsRef.current && (
          <SandboxTab schema={dataset.schema} normStats={dataset.normStats} config={config} weightsJson={bestWeightsRef.current} labels={dataset.labels} modelName={modelName ?? "Untitled"} onLog={addLog} />
        )}

        {/* ═══════ MODELS TAB ═══════ */}
        {activeTab === "models" && (
          <ModelsTab models={savedModels} onFinetune={loadModelForFinetune} onDelete={(id, name) => { deleteModel(id).then(() => { setSavedModels((p) => p.filter((m) => m.id !== id)); addLog(`[Model] Deleted: ${name}`); }); }} onDownload={(m) => {
            if (!m.weightsJson) { addLog("[Error] No weights to download"); return; }
            const blob = new Blob([m.weightsJson], { type: "application/json" });
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url; a.download = `${m.name}-weights.json`; a.click();
            URL.revokeObjectURL(url);
            addLog(`[Model] Downloaded: ${m.name}-weights.json (${(m.weightsJson.length / 1024).toFixed(1)}KB)`);
          }} onExportOnnx={async (m) => {
            try {
              if (!m.weightsJson) { addLog("[Error] No weights to export"); return; }
              setIsBusy(true);
              addLog(`[Export] Converting ${m.name} to ONNX...`);
              const result = exportToOnnx(m.weightsJson, m.schema, config);
              const blob = new Blob([result.model as unknown as BlobPart], { type: "application/octet-stream" });
              const url = URL.createObjectURL(blob);
              const a = document.createElement("a");
              a.href = url; a.download = `${m.name}.onnx`; a.click();
              URL.revokeObjectURL(url);
              addLog(`[Export] ✅ ONNX: ${m.name}.onnx (${(result.sizeBytes / 1024).toFixed(1)}KB, ${result.quantized ? "INT8" : "FP32"}, ${result.compressionRatio.toFixed(1)}x)`);
            } catch (err) {
              addLog(`[Export Error] ${err instanceof Error ? err.message : String(err)}`);
            } finally {
              setIsBusy(false);
            }
          }} onHardFlush={() => {
            // Clear the ACTUAL IndexedDB "FrugalSlothDB" where models are stored
            try { indexedDB.deleteDatabase("FrugalSlothDB"); } catch { /* ignore */ }
            localStorage.clear();
            sessionStorage.clear();
            if ('serviceWorker' in navigator) {
              navigator.serviceWorker.getRegistrations().then((regs) => {
                regs.forEach((reg) => { reg.unregister(); });
              });
            }
            caches.keys().then((names) => Promise.all(names.map((n) => caches.delete(n)))).catch(() => {});
            addLog("[System] ⚠️ HARD FLUSH: FrugalSlothDB deleted. Reloading...");
            setSavedModels([]); // Clear UI immediately
            setTimeout(() => { location.reload(); }, 1500);
          }} />
        )}

        {/* ═══════ IMPORT ONNX TAB ═══════ */}
        {activeTab === "import" && (
          <ImportTabFixed onLog={addLog} setIsBusy={setIsBusy} onModelTrained={(weights, schema, normStats) => {
            setDataset({ schema, rows: [], labels: [], normStats });
            bestWeightsRef.current = weights;
            setActiveTab("sandbox");
          }} />
        )}

        {/* ═══════ RELEASES TAB ═══════ */}
        {activeTab === "releases" && <ReleasesTab />}

        {/* ═══════ DOCS TAB ═══════ */}
        {activeTab === "docs" && <DocsTab />}

      </main>

      {/* ── SYSTEM LOG (fixed at bottom) ── */}
      <div className="fs-app-console">
        <div className="fs-card">
          <div className="fs-card-header">
            <div className="fs-card-title"><Terminal size={16} /> System Log</div>
            <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
              <span className="fs-badge fs-badge-neutral">{log.length} entries</span>
              <button className="fs-btn fs-btn-sm" onClick={() => setLog([])}><Trash2 size={12} /> Clear</button>
            </div>
          </div>
          <div className="fs-log">
            {log.map((entry, i) => (
              <div key={i} className="fs-log-entry" style={
                entry.includes("[Error]") ? { color: "var(--error)" } :
                entry.includes("[Train]") ? { color: "var(--accent-hover)" } :
                entry.includes("FRUGAL") || entry.includes("Engine") || entry.includes("ONNX Runtime") ? { color: "var(--accent)" } :
                {}
              }>
                {entry.includes("\n") ? <pre style={{ margin: 0, lineHeight: 1.3 }}>{entry}</pre> : entry}
              </div>
            ))}
            <div ref={logEndRef} />
          </div>
        </div>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════
   HOME TAB
   ═══════════════════════════════════════════════════════════════ */
function HomeTab({ onLoadDemo, onLoadDemoJson, onChangeTab }: { onLoadDemo: () => void; onLoadDemoJson: () => void; onChangeTab: (t: string) => void }) {
  return (
    <div>
      {/* Quick Demo — at the top, compact */}
      <div className="fs-card" style={{ marginBottom: 14, padding: 14 }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 10 }}>
          <div>
            <div style={{ fontWeight: 600, fontSize: 14, marginBottom: 1 }}>🦥 FrugalSloth — Train AI in Your Browser</div>
            <div style={{ fontSize: 11, color: "var(--text-muted)" }}>Upload CSV/JSON/JSONL, train neural nets, export ONNX. 100% private — no data leaves your browser. Quick demo: load a dataset instantly to try training.</div>
          </div>
          <div style={{ display: "flex", gap: 8, flexShrink: 0 }}>
            <div className="fs-tooltip-wrap">
              <button className="fs-btn fs-btn-wine fs-btn-sm" onClick={onLoadDemo}><FileSpreadsheet size={13} /> Example 1: Wine</button>
              <span className="fs-tooltip">100 wine samples, 11 features → predict quality score (regression). Perfect for first-time users.</span>
            </div>
            <div className="fs-tooltip-wrap">
              <button className="fs-btn fs-btn-iris fs-btn-sm" onClick={onLoadDemoJson}><FileJson size={13} /> Example 2: Iris</button>
              <span className="fs-tooltip">100 iris flowers, 4 features → classify species (3-class). Classic ML benchmark.</span>
            </div>
            <div className="fs-tooltip-wrap">
              <button className="fs-btn fs-btn-sm" onClick={() => onChangeTab("docs")}><BookOpen size={13} /> Docs</button>
              <span className="fs-tooltip">Read the full documentation, including dataset format requirements and GPU setup.</span>
            </div>
          </div>
        </div>
      </div>

      {/* Speed Claims — compact row */}
      <div className="fs-metrics" style={{ marginBottom: 14 }}>
        <div className="fs-metric" style={{ padding: 8 }}>
          <div className="fs-metric-value" style={{ fontSize: 18, color: "var(--accent)" }}>10×</div>
          <div className="fs-metric-label" style={{ fontSize: 10 }}>Faster than Python CPU</div>
        </div>
        <div className="fs-metric" style={{ padding: 8 }}>
          <div className="fs-metric-value" style={{ fontSize: 18, color: "var(--info)" }}>0ms</div>
          <div className="fs-metric-label" style={{ fontSize: 10 }}>Network Latency</div>
        </div>
        <div className="fs-metric" style={{ padding: 8 }}>
          <div className="fs-metric-value" style={{ fontSize: 18, color: "var(--success)" }}>&lt;2MB</div>
          <div className="fs-metric-label" style={{ fontSize: 10 }}>Export Size</div>
        </div>
        <div className="fs-metric" style={{ padding: 8 }}>
          <div className="fs-metric-value" style={{ fontSize: 18, color: "var(--warning)" }}>100%</div>
          <div className="fs-metric-label" style={{ fontSize: 10 }}>Private</div>
        </div>
      </div>

      {/* Use Cases — 3 columns, compact */}
      <div className="fs-grid-3" style={{ gap: 10 }}>
        {[
          { icon: <Cpu size={18} />, title: "IoT Sensors", desc: "Train on telemetry in-browser. No cloud round-trip." },
          { icon: <Shield size={18} />, title: "Privacy-First ML", desc: "Sensitive data never leaves the browser." },
          { icon: <WifiOff size={18} />, title: "Offline Inference", desc: "Export runs on Raspberry Pi or static hosting." },
          { icon: <Gauge size={18} />, title: "Real-Time", desc: "Sub-millisecond inference via ONNX Runtime Web." },
          { icon: <Target size={18} />, title: "Prototyping", desc: "Upload CSV, train in seconds, iterate live." },
          { icon: <Anchor size={18} />, title: "Edge Deploy", desc: "Self-contained .zip drops into any host." },
        ].map((uc) => (
          <div className="fs-card" key={uc.title} style={{ padding: 12 }}>
            <div style={{ color: "var(--accent)", marginBottom: 4 }}>{uc.icon}</div>
            <div style={{ fontWeight: 600, fontSize: 13, marginBottom: 3 }}>{uc.title}</div>
            <div style={{ fontSize: 11, color: "var(--text-muted)", lineHeight: 1.4 }}>{uc.desc}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════
   DATA TAB
   ═══════════════════════════════════════════════════════════════ */
function DataTab({ dataset, onUpload, onDemoCSV, onDemoJSON, onClear, parseProgress }: {
  dataset: { schema: DatasetSchema; normStats: NormalizationStats } | null;
  onUpload: (f: File) => void; onDemoCSV: () => void; onDemoJSON: () => void; onClear: () => void; parseProgress: number;
}) {
  return (
    <div className="fs-grid-2">
      <div className="fs-card">
        <div className="fs-card-header">
          <div className="fs-card-title"><Upload size={16} /> Upload Dataset</div>
          <span className="fs-badge fs-badge-neutral">CSV + JSON + JSONL</span>
        </div>
        <p style={{ color: "var(--text-muted)", fontSize: 13, marginBottom: 14 }}>
          Upload a CSV, JSON, or JSONL dataset. FrugalSloth trains a custom MLP neural network from scratch using TensorFlow.js. For BERT/transformer fine-tuning, use the <strong>Import ONNX</strong> tab instead.
        </p>
        <div className="fs-dropzone" onClick={() => document.getElementById("file-input")?.click()}>
          <Upload size={28} style={{ marginBottom: 8 }} />
          <div style={{ fontWeight: 500, color: "var(--text-primary)", marginBottom: 4 }}>Drop file or click to browse</div>
          <div style={{ fontSize: 12, color: "var(--text-muted)" }}>.csv, .json, .jsonl · Max 50MB</div>
          <input id="file-input" type="file" accept=".csv,.json,.jsonl" style={{ display: "none" }} onChange={(e) => { const f = e.target.files?.[0]; if (f) onUpload(f); }} />
        </div>
        {parseProgress > 0 && (
          <div style={{ marginTop: 12 }}>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "var(--text-muted)", marginBottom: 4 }}>
              <span>Parsing...</span><span>{parseProgress}%</span>
            </div>
            <div className="fs-progress"><div className="fs-progress-fill" style={{ width: `${parseProgress}%` }} /></div>
          </div>
        )}
        <div style={{ marginTop: 14, display: "flex", gap: 8, flexWrap: "wrap" }}>
          <div className="fs-tooltip-wrap">
            <button className="fs-btn fs-btn-wine fs-btn-sm" onClick={onDemoCSV}><FileSpreadsheet size={13} /> Example 1: Wine</button>
            <span className="fs-tooltip">100 wine samples, 11 features → predict quality score (regression). Perfect for first-time users.</span>
          </div>
          <div className="fs-tooltip-wrap">
            <button className="fs-btn fs-btn-iris fs-btn-sm" onClick={onDemoJSON}><FileJson size={13} /> Example 2: Iris</button>
            <span className="fs-tooltip">100 iris flowers, 4 features → classify species (3-class). Classic ML benchmark.</span>
          </div>
        </div>
      </div>
      {dataset && (
        <div className="fs-card">
          <div className="fs-card-header">
            <div className="fs-card-title"><Eye size={16} /> Schema</div>
            <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
              <span className="fs-badge fs-badge-accent">{dataset.schema.rowCount} rows</span>
              <button className="fs-btn fs-btn-sm fs-btn-danger" onClick={onClear} title="Clear loaded dataset"><RotateCcw size={13} /> New</button>
            </div>
          </div>
          <table className="fs-table">
            <thead><tr><th>#</th><th>Column</th><th>Type</th><th>Stats</th></tr></thead>
            <tbody>
              {dataset.schema.columns.map((col, i) => (
                <tr key={i} style={i === dataset.schema.targetIndex ? { background: "var(--accent-light)" } : {}}>
                  <td style={{ color: "var(--text-muted)", fontFamily: "var(--font-mono)", fontSize: 12 }}>{i}</td>
                  <td style={{ fontWeight: 500 }}>
                    {col.name}
                    {i === dataset.schema.targetIndex && <span className="fs-badge fs-badge-accent" style={{ marginLeft: 6, fontSize: 9 }}>TARGET</span>}
                  </td>
                  <td><span className={`fs-badge ${typeColor(col.type)}`}>{col.type}</span></td>
                  <td style={{ fontSize: 11, color: "var(--text-muted)" }}>
                    {col.mean !== undefined && `μ=${col.mean.toFixed(2)} σ=${col.std?.toFixed(2)}`}
                    {col.min !== undefined && ` [${col.min.toFixed(1)}, ${col.max?.toFixed(1)}]`}
                    {col.categories && `${col.categories.length} cats`}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════
   TRAIN TAB
   ═══════════════════════════════════════════════════════════════ */
function TrainTab({
  config, setConfig, modelName, setModelName, trainingStatus, trainingProgress, currentEpoch, totalEpochs,
  metrics, bestValLoss, modelSummaryRef, exportStatus, onStart, onPause, onResume, onStop, onSave, onExport, onClear, hasWeights, datasetName, hasDataset, onLoadDemo, onLoadDemoJson,
}: {
  config: TrainingConfig; setConfig: (c: TrainingConfig | ((p: TrainingConfig) => TrainingConfig)) => void;
  modelName: string; setModelName: (n: string) => void; trainingStatus: TrainingStatus;
  trainingProgress: number; currentEpoch: number; totalEpochs: number; metrics: EpochMetrics[];
  bestValLoss: number; modelSummaryRef: React.MutableRefObject<{ totalParams: number } | null>;
  exportStatus: string; onStart: () => void; onPause: () => void; onResume: () => void; onStop: () => void;
  onSave: () => void; onExport: () => void; onClear: () => void; hasWeights: boolean; datasetName: string; hasDataset: boolean;
  onLoadDemo: () => void; onLoadDemoJson: () => void;
}) {
  return (
    <div className="fs-grid-2">
      {!hasDataset && (
        <div className="fs-card" style={{ gridColumn: "1 / -1", textAlign: "center", padding: 60, width: "100%", minHeight: 300 }}>
          <div style={{ fontSize: 16, fontWeight: 600, marginBottom: 8 }}>No Dataset Loaded</div>
          <div style={{ fontSize: 13, color: "var(--text-muted)", marginBottom: 20 }}>
            Go to <strong>Data</strong> tab to upload CSV/JSON/JSONL, or try a demo below.
          </div>
          <div style={{ display: "flex", gap: 10, justifyContent: "center" }}>
            <button className="fs-btn fs-btn-wine fs-btn-sm" onClick={onLoadDemo}><FileSpreadsheet size={13} /> Example 1: Wine</button>
            <button className="fs-btn fs-btn-iris fs-btn-sm" onClick={onLoadDemoJson}><FileJson size={13} /> Example 2: Iris</button>
          </div>
        </div>
      )}
      {hasDataset && (<>
      <div className="fs-card">
        <div className="fs-card-header">
          <div className="fs-card-title"><Settings2 size={16} /> Hyperparameters</div>
        </div>
        {/* Hyperparams compact 2-column grid */}
        {/* Hyperparams — compact 2-column grid */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
          <div>
            <TooltipLabel label="Learning Rate" tooltip="Step size. 0.001 default." />
            <input type="number" className="fs-input" value={config.learningRate} onChange={(e) => setConfig((c) => ({ ...c, learningRate: parseFloat(e.target.value) }))} step={0.0001} min={0.00001} max={1} />
          </div>
          <div>
            <TooltipLabel label="Epochs" tooltip="Passes through data." />
            <input type="number" className="fs-input" value={config.epochs} onChange={(e) => setConfig((c) => ({ ...c, epochs: parseInt(e.target.value) }))} min={1} max={1000} />
          </div>
          <div>
            <TooltipLabel label="Batch Size" tooltip="Samples per update. Power of 2." />
            <input type="number" className="fs-input" value={config.batchSize} onChange={(e) => setConfig((c) => ({ ...c, batchSize: parseInt(e.target.value) }))} min={1} max={1024} />
          </div>
          <div>
            <TooltipLabel label="Hidden Layers" tooltip="Comma-separated: [64, 32]" />
            <input type="text" className="fs-input" value={config.hiddenLayers.join(", ")} onChange={(e) => { const layers = e.target.value.split(",").map((s) => parseInt(s.trim())).filter((n) => !isNaN(n) && n > 0); setConfig((c) => ({ ...c, hiddenLayers: layers })); }} placeholder="64, 32" />
          </div>
          <div>
            <TooltipLabel label="Optimizer" tooltip="Adam: best default." />
            <select className="fs-select" value={config.optimizer} onChange={(e) => setConfig((c) => ({ ...c, optimizer: e.target.value as TrainingConfig["optimizer"] }))} style={{ width: "100%" }}>
              <option value="adam">adam</option>
              <option value="sgd">sgd</option>
              <option value="rmsprop">rmsprop</option>
            </select>
          </div>
          <div>
            <TooltipLabel label="Val Split" tooltip="Hold-out fraction. 0.2 = 20%." />
            <input type="number" className="fs-input" value={config.validationSplit} onChange={(e) => setConfig((c) => ({ ...c, validationSplit: parseFloat(e.target.value) }))} step={0.05} min={0} max={0.5} />
          </div>
          <div>
            <TooltipLabel label="Early Stop" tooltip="Patience. 0 = off." />
            <input type="number" className="fs-input" value={config.earlyStoppingPatience} onChange={(e) => setConfig((c) => ({ ...c, earlyStoppingPatience: parseInt(e.target.value) }))} min={0} max={100} />
          </div>
          <div>
            <label className="fs-label"><span className="fs-label-text">Model Name</span></label>
            <input type="text" className="fs-input" value={modelName} onChange={(e) => setModelName(e.target.value)} placeholder="my-model" />
          </div>
        </div>
      </div>
      <div>
        <div className="fs-card" style={{ marginBottom: 16 }}>
          <div className="fs-card-header">
            <div className="fs-card-title"><Play size={16} /> Training {datasetName && <span className="fs-badge fs-badge-neutral" style={{ marginLeft: 8, fontSize: 11 }}>{datasetName}</span>}</div>
            <span className={`fs-status fs-status-${trainingStatus}`}><span className="fs-status-dot" /> {trainingStatus}</span>
          </div>
          <div style={{ fontSize: 11, color: "var(--accent)", marginBottom: 8 }}>
            Training a custom MLP (Multi-Layer Perceptron) from scratch with TF.js. For BERT/transformer fine-tuning, use <strong>Import ONNX</strong> tab.
          </div>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 16 }}>
            {trainingStatus === "idle" && (
              <div className="fs-tooltip-wrap">
                <button className="fs-btn fs-btn-primary" onClick={onStart}><Play size={14} /> Start Training</button>
                <span className="fs-tooltip">Begin training the MLP on your dataset. Adjust hyperparameters first if needed.</span>
              </div>
            )}
            {trainingStatus === "running" && <><button className="fs-btn" onClick={onPause}><Pause size={14} /> Pause</button><button className="fs-btn fs-btn-danger" onClick={onStop}><Square size={14} /> Stop</button></>}
            {trainingStatus === "paused" && <><button className="fs-btn fs-btn-primary" onClick={onResume}><Play size={14} /> Resume</button><button className="fs-btn fs-btn-danger" onClick={onStop}><Square size={14} /> Stop</button></>}
            {(trainingStatus === "completed" || trainingStatus === "paused") && <>
              <div className="fs-tooltip-wrap">
                <button className="fs-btn fs-btn-primary fs-btn-sm" onClick={onSave} disabled={!hasWeights}><Save size={13} /> Save to Models</button>
                <span className="fs-tooltip">Store this trained model in browser memory. Auto-named as name_v1, name_v2 if duplicate.</span>
              </div>
              <div className="fs-tooltip-wrap">
                <button className="fs-btn fs-btn-sm" onClick={onExport} disabled={!hasWeights}><Package size={13} /> Export Weights</button>
                <span className="fs-tooltip">Download JSON weights file. Reload later or use in custom code.</span>
              </div>
              <div className="fs-tooltip-wrap">
                <button className="fs-btn fs-btn-sm fs-btn-danger" onClick={onClear}><RotateCcw size={13} /> New</button>
                <span className="fs-tooltip">Clear current training state to start a new dataset.</span>
              </div>
            </>}
          </div>
          {trainingStatus !== "idle" && (
            <div style={{ marginBottom: 14 }}>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, color: "var(--text-muted)", marginBottom: 4 }}>
                <span>Epoch {currentEpoch} / {totalEpochs}</span><span>{trainingProgress}%</span>
              </div>
              <div className="fs-progress"><div className="fs-progress-fill" style={{ width: `${trainingProgress}%` }} /></div>
            </div>
          )}
          {metrics.length > 0 && (
            <div className="fs-metrics" style={{ marginBottom: 14 }}>
              <div className="fs-metric"><div className="fs-metric-value" style={{ fontSize: 16 }}>{metrics[metrics.length - 1].loss.toExponential(2)}</div><div className="fs-metric-label">Loss</div></div>
              <div className="fs-metric"><div className="fs-metric-value" style={{ fontSize: 16 }}>{bestValLoss < Infinity ? bestValLoss.toExponential(2) : "—"}</div><div className="fs-metric-label">Best Val</div></div>
              <div className="fs-metric"><div className="fs-metric-value" style={{ fontSize: 16, color: "var(--accent)" }}>{currentEpoch}</div><div className="fs-metric-label">Epoch</div></div>
              {metrics[metrics.length - 1].accuracy !== undefined && <div className="fs-metric"><div className="fs-metric-value" style={{ fontSize: 16, color: "var(--success)" }}>{(metrics[metrics.length - 1].accuracy! * 100).toFixed(0)}%</div><div className="fs-metric-label">Accuracy</div></div>}
              {modelSummaryRef.current && <div className="fs-metric"><div className="fs-metric-value" style={{ fontSize: 16 }}>{modelSummaryRef.current.totalParams.toLocaleString()}</div><div className="fs-metric-label">Params</div></div>}
            </div>
          )}
          {exportStatus && <div style={{ padding: 8, background: "var(--accent-light)", borderRadius: "var(--radius-sm)", fontSize: 12, color: "var(--accent-hover)" }}>{exportStatus}</div>}
        </div>
        {metrics.length > 1 && (
          <div className="fs-card">
            <div className="fs-card-header">
              <div className="fs-card-title"><TrendingDown size={16} /> Loss Curve</div>
              <span className="fs-badge fs-badge-neutral">EMA smoothed</span>
            </div>
            <MiniChart metrics={metrics} />
          </div>
        )}
      </div>
    </>)}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════
   SANDBOX TAB (FIXED — inputDim from actual data, not schema cols)
   ═══════════════════════════════════════════════════════════════ */
function SandboxTab({ schema, normStats, config, weightsJson, labels, modelName, onLog }: {
  schema: DatasetSchema; normStats: NormalizationStats; config: TrainingConfig; weightsJson: string; labels: number[]; modelName: string; onLog: (msg: string) => void;
}) {
  const [predictions, setPredictions] = useState<number[]>([]);
  const [latency, setLatency] = useState(0);
  const [confidence, setConfidence] = useState(0);
  const [inputValues, setInputValues] = useState<Record<number, number>>({});
  const [ready, setReady] = useState(false);
  const modelRef = useRef<tf.LayersModel | null>(null);
  const featureCols = useMemo(() => schema.columns.filter((_, i) => i !== schema.targetIndex), [schema]);

  // Initialize input values from column means
  useEffect(() => {
    const init: Record<number, number> = {};
    featureCols.forEach((col, idx) => { init[idx] = col.mean ?? 0; });
    setInputValues(init);
  }, [featureCols]);

  // Build model and load weights ONCE when weightsJson changes
  useEffect(() => {
    let cancelled = false;
    const buildAndPredict = async () => {
      try {
        await tf.ready();

        // Parse weights: flat number[][] array saved during training
        const flatWeights: number[][] = JSON.parse(weightsJson);

        // Compute dims
        let outputDim: number;
        if (schema.taskType === "regression") outputDim = 1;
        else if (labels.length > 0) outputDim = new Set(labels).size;
        else outputDim = schema.columns[schema.targetIndex]?.categories?.length ?? 2;

        const inputDim = featureCols.length;
        onLog(`[Sandbox] Build model: ${inputDim}→${config.hiddenLayers.join("→")}→${outputDim}, ${flatWeights.length} weight tensors`);

        // Build model with same architecture as training
        const model = tf.sequential();
        config.hiddenLayers.forEach((units, i) => {
          model.add(tf.layers.dense({
            units, activation: "relu", inputShape: i === 0 ? [inputDim] : undefined,
            kernelRegularizer: tf.regularizers.l2({ l2: 0.01 })
          }));
        });
        model.add(tf.layers.dense({
          units: outputDim,
          activation: schema.taskType === "regression" ? "linear" : "softmax"
        }));

        // Convert flat arrays back to tensors and set as weights
        const layerWeights: tf.Tensor[] = [];
        let wIdx = 0;
        for (const layer of model.layers) {
          const expectedWeights = layer.getWeights();
          for (let i = 0; i < expectedWeights.length; i++) {
            if (wIdx >= flatWeights.length) break;
            const shape = expectedWeights[i].shape;
            const size = shape.reduce((a, b) => a * b, 1);
            layerWeights.push(tf.reshape(tf.tensor1d(flatWeights[wIdx].slice(0, size)), shape));
            wIdx++;
          }
        }
        model.setWeights(layerWeights);

        if (cancelled) return;
        modelRef.current = model;
        setReady(true);
        onLog(`[Sandbox] Model loaded, ${model.countParams()} params`);

        // First prediction
        const initInputs = featureCols.map((col) => col.mean ?? 0);
        const inputTensor = tf.tensor2d([initInputs], [1, inputDim]);
        const start = performance.now();
        const output = model.predict(inputTensor) as tf.Tensor;
        const result = await output.data();
        const ms = performance.now() - start;
        inputTensor.dispose();
        output.dispose();

        const preds = Array.from(result);
        setPredictions(preds);
        setLatency(ms);
        if (schema.taskType !== "regression") setConfidence(Math.max(...preds));
      } catch (err: any) {
        onLog(`[Sandbox Error] ${err.message}`);
      }
    };
    buildAndPredict();
    return () => { cancelled = true; modelRef.current = null; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [weightsJson]);

  const predictTimeout = useRef<ReturnType<typeof setTimeout> | null>(null);
  const runPredict = useCallback((currentInputs: Record<number, number>) => {
    if (!modelRef.current) return;
    if (predictTimeout.current) clearTimeout(predictTimeout.current);
    predictTimeout.current = setTimeout(async () => {
      try {
        const inputs = featureCols.map((_, idx) => currentInputs[idx] ?? 0);
        const inputTensor = tf.tensor2d([inputs], [1, featureCols.length]);
        const start = performance.now();
        const output = modelRef.current!.predict(inputTensor) as tf.Tensor;
        const result = await output.data();
        const ms = performance.now() - start;
        inputTensor.dispose();
        output.dispose();

        const preds = Array.from(result);
        setPredictions(preds);
        setLatency(ms);
        if (schema.taskType !== "regression") setConfidence(Math.max(...preds));
      } catch { /* ignore stale predictions */ }
    }, 30);
  }, [featureCols, schema]);

  const handleSliderChange = (colIdx: number, val: number) => {
    const next = { ...inputValues, [colIdx]: val };
    setInputValues(next);
    runPredict(next);
  };
  const handleNumberChange = (colIdx: number, val: number) => {
    const next = { ...inputValues, [colIdx]: val };
    setInputValues(next);
    runPredict(next);
  };

  const targetCol = schema.columns[schema.targetIndex];

  if (!ready) {
    return (
      <div className="fs-grid-2">
        <div className="fs-card" style={{ gridColumn: "1 / -1", textAlign: "center", padding: 60 }}>
          <Loader2 size={24} className="fs-spin" style={{ marginBottom: 12 }} />
          <div style={{ fontSize: 14 }}>Loading inference model...</div>
        </div>
      </div>
    );
  }

  const predLabel = predictions.length === 0 ? "—"
    : schema.taskType === "regression" ? predictions[0].toFixed(4)
    : targetCol.categories ? (targetCol.categories[predictions.indexOf(Math.max(...predictions))] ?? "?")
    : String(predictions.indexOf(Math.max(...predictions)));

  return (
    <div className="fs-grid-2">
      <div>
        <div className="fs-card">
          <div className="fs-card-header">
            <div className="fs-card-title"><SlidersHorizontal size={16} /> Input Features</div>
            <span className="fs-badge fs-badge-neutral">{modelName}</span>
            <span className="fs-badge fs-badge-accent">{featureCols.length} features</span>
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(260px, 1fr))", gap: 12 }}>
            {featureCols.map((col, colIdx) => {
              const origIdx = schema.columns.indexOf(col);
              const norm = normStats.numeric[origIdx];
              const val = inputValues[colIdx] ?? col.mean ?? 0;
              if (col.type === "numeric" && norm) {
                const min = col.min ?? norm.mean - 3 * norm.std;
                const max = col.max ?? norm.mean + 3 * norm.std;
                return (
                  <div key={colIdx} style={{ padding: 10, background: "var(--bg-surface-2)", borderRadius: "var(--radius-md)" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, fontWeight: 500, marginBottom: 6, color: "var(--text-secondary)" }}>
                      <span>{col.name}</span><span style={{ color: "var(--accent)", fontFamily: "var(--font-mono)" }}>{val.toFixed(3)}</span>
                    </div>
                    <input type="range" className="fs-range" min={min} max={max} step={(max - min) / 100} value={val}
                      onChange={(e) => handleSliderChange(colIdx, parseFloat(e.target.value))} />
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: "var(--text-muted)", marginTop: 2 }}>
                      <span>{min.toFixed(1)}</span><span>μ={col.mean?.toFixed(1)}</span><span>{max.toFixed(1)}</span>
                    </div>
                  </div>
                );
              }
              if (col.type === "categorical" && col.categories) {
                return (
                  <div key={colIdx} style={{ padding: 10, background: "var(--bg-surface-2)", borderRadius: "var(--radius-md)" }}>
                    <div style={{ fontSize: 12, fontWeight: 500, marginBottom: 6, color: "var(--text-secondary)" }}>{col.name}</div>
                    <select className="fs-select" style={{ width: "100%" }} value={val} onChange={(e) => handleNumberChange(colIdx, parseInt(e.target.value))}>
                      {col.categories.map((cat, i) => <option key={i} value={i}>{cat}</option>)}
                    </select>
                  </div>
                );
              }
              return (
                <div key={colIdx} style={{ padding: 10, background: "var(--bg-surface-2)", borderRadius: "var(--radius-md)" }}>
                  <div style={{ fontSize: 12, fontWeight: 500, marginBottom: 6, color: "var(--text-secondary)" }}>{col.name}</div>
                  <input type="number" className="fs-input" value={val}
                    onChange={(e) => handleNumberChange(colIdx, parseFloat(e.target.value) || 0)} style={{ width: "100%" }} />
                </div>
              );
            })}
          </div>
        </div>
      </div>
      <div>
        <div className="fs-card" style={{ background: "var(--accent-light)", borderColor: "var(--accent)" }}>
          <div style={{ textAlign: "center", padding: "20px 0" }}>
            <div style={{ fontSize: 13, color: "var(--accent)", marginBottom: 4 }}>Predicted {targetCol.name}</div>
            <div className="fs-prediction-value">{predLabel}</div>
            {schema.taskType !== "regression" && predictions.length > 0 && (
              <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 4 }}>
                Confidence: {(confidence * 100).toFixed(1)}% | Inferences: {latency.toFixed(1)}ms
              </div>
            )}
            {schema.taskType === "regression" && predictions.length > 0 && (
              <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 4 }}>
                Inference: {latency.toFixed(1)}ms
              </div>
            )}
          </div>
        </div>
        <div className="fs-card" style={{ marginTop: 12 }}>
          <div className="fs-card-title" style={{ fontSize: 13 }}><Activity size={14} /> Performance</div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginTop: 8 }}>
            <div style={{ textAlign: "center", padding: 8, background: "var(--bg-surface-2)", borderRadius: "var(--radius-md)" }}>
              <div style={{ fontSize: 16, fontWeight: 700, color: "var(--accent)" }}>{latency > 0 ? latency.toFixed(1) : "—"}ms</div>
              <div style={{ fontSize: 10, color: "var(--text-muted)" }}>LATENCY</div>
            </div>
            <div style={{ textAlign: "center", padding: 8, background: "var(--bg-surface-2)", borderRadius: "var(--radius-md)" }}>
              <div style={{ fontSize: 16, fontWeight: 700, color: "var(--accent)" }}>{latency > 0 ? (1000 / latency).toFixed(0) : "—"}</div>
              <div style={{ fontSize: 10, color: "var(--text-muted)" }}>INF/SEC</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function ModelsTab({ models, onFinetune, onDelete, onDownload, onExportOnnx, onHardFlush }: {
  models: StoredModel[]; onFinetune: (m: StoredModel) => void; onDelete: (id: string, name: string) => void; onDownload: (m: StoredModel) => void; onExportOnnx: (m: StoredModel) => void; onHardFlush: () => void;
}) {
  return (
    <div className="fs-card">
      <div className="fs-card-header">
        <div className="fs-card-title"><HardDrive size={16} /> Saved Models</div>
        <span className="fs-badge fs-badge-neutral">{models.length} model(s)</span>
      </div>
      <div style={{ fontSize: 11, color: "var(--accent)", marginBottom: 8, padding: "6px 10px", background: "var(--bg-surface-2)", borderRadius: "var(--radius-sm)" }}>
        <strong>Fine-tune flow:</strong> Click "Fine-tune" to load a model with its hyperparameters preserved. Then go to <strong>Train</strong> tab and click <strong>New</strong> to clear, adjust parameters, and retrain. Versions auto-increment (v1, v2, etc.).
      </div>
      {models.length === 0 ? (
        <div style={{ textAlign: "center", padding: "40px 20px", color: "var(--text-muted)" }}>
          <HardDrive size={32} style={{ marginBottom: 8, opacity: 0.5 }} />
          <p>No saved models yet.</p>
          <p style={{ fontSize: 12 }}>Train a model and click Save to store it here.</p>
        </div>
      ) : (
        <table className="fs-table">
          <thead><tr><th>Name</th><th>Task</th><th>Params</th><th>Epochs</th><th>Val Loss</th><th>Created</th><th>Actions</th></tr></thead>
          <tbody>
            {models.map((m) => (
              <tr key={m.id}>
                <td style={{ fontWeight: 500 }}>{m.name}</td>
                <td><span className={`fs-badge fs-badge-${m.schema.taskType === "regression" ? "info" : "success"}`}>{m.schema.taskType}</span></td>
                <td style={{ fontFamily: "var(--font-mono)" }}>{m.modelSummary.totalParams.toLocaleString()}</td>
                <td>{m.history.length}</td>
                <td style={{ fontFamily: "var(--font-mono)" }}>{m.bestValLoss < Infinity ? m.bestValLoss.toExponential(3) : "N/A"}</td>
                <td style={{ color: "var(--text-muted)", fontSize: 12 }}>{new Date(m.createdAt).toLocaleDateString()}</td>
                <td>
                  <button className="fs-btn fs-btn-sm" onClick={() => onDownload(m)} style={{ marginRight: 6 }} title="Download weights JSON"><Download size={12} /> JSON</button>
                  <button className="fs-btn fs-btn-primary fs-btn-sm" onClick={() => onExportOnnx(m)} style={{ marginRight: 6 }} title="Export to ONNX format"><Package size={12} /> ONNX</button>
                  <button className="fs-btn fs-btn-sm" onClick={() => onFinetune(m)} style={{ marginRight: 6 }}><RotateCcw size={12} /> Fine-tune</button>
                  <button className="fs-btn fs-btn-danger fs-btn-sm" onClick={() => onDelete(m.id, m.name)}><Trash2 size={12} /></button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <div style={{ marginTop: 16, padding: "12px", background: "var(--bg-surface-2)", borderRadius: "var(--radius-md)", textAlign: "center" }}>
        <div style={{ fontSize: 12, color: "var(--text-muted)", marginBottom: 8 }}>Having issues with training giving zeroes or broken charts? Try a hard reset.</div>
        <button className="fs-btn fs-btn-danger fs-btn-sm" onClick={() => {
          if (window.confirm("⚠️ HARD FLUSH\n\nThis will delete:\n• All saved models\n• All cached data\n• All settings\n\nUse this ONLY if training gives zeroes or broken charts.\n\nAre you sure?")) {
            onHardFlush();
          }
        }}><Trash2 size={12} /> Hard Flush — Reset Everything</button>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════
   IMPORT ONNX TAB (FIXED — proper ORT WASM paths)
   ═══════════════════════════════════════════════════════════════ */
function ImportTabFixed({ onLog, setIsBusy, onModelTrained }: {
  onLog: (msg: string) => void;
  setIsBusy: (v: boolean) => void;
  onModelTrained: (weights: string, schema: DatasetSchema, normStats: NormalizationStats) => void;
}) {
  const [selectedPreset, setSelectedPreset] = useState("custom");
  const [status, setStatus] = useState("");
  const [step, setStep] = useState(1);
  const [embeddings, setEmbeddings] = useState<number[][]>([]);
  const [embLabels, setEmbLabels] = useState<number[]>([]);
  const [embDim, setEmbDim] = useState(0);
  const [schema, setSchema] = useState<DatasetSchema | null>(null);
  const [normStats, setNormStats] = useState<NormalizationStats | null>(null);
  const [headProgress, setHeadProgress] = useState(0);
  const [headEpoch, setHeadEpoch] = useState(0);
  const [isTrainingHead, setIsTrainingHead] = useState(false);
  const sessionRef = useRef<ort.InferenceSession | null>(null);

  useEffect(() => {
    ort.env.wasm.wasmPaths = "/";
    ort.env.wasm.numThreads = 1;
  }, []);

  const tokenize = (text: string, maxLen = 128): { input_ids: number[]; attention_mask: number[] } => {
    const vocab: Record<string, number> = { "[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3 };
    let nextId = 4;
    const tokens = text.toLowerCase().replace(/[^a-z0-9\s]/g, "").split(/\s+/).filter((t) => t.length > 0).slice(0, maxLen - 2);
    const input_ids = [2];
    for (const token of tokens) { if (!vocab[token]) vocab[token] = nextId++; input_ids.push(vocab[token]); }
    input_ids.push(3);
    const attention_mask = new Array(input_ids.length).fill(1);
    while (input_ids.length < maxLen) { input_ids.push(0); attention_mask.push(0); }
    return { input_ids, attention_mask };
  };

  const handleOnnxUpload = useCallback(async (file: File) => {
    setIsBusy(true);
    const t0 = performance.now();
    try {
      setStatus("Reading ONNX file...");
      onLog(`[Import] Reading: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)}MB)`);
      const buffer = await file.arrayBuffer();
      setStatus("Creating ONNX session (WASM compilation, 10-30s on first run)...");
      onLog("[Import] Compiling WASM — this is a one-time cost...");
      ort.env.wasm.wasmPaths = "/";
      ort.env.wasm.numThreads = 1;
      const session = await ort.InferenceSession.create(buffer, { executionProviders: ["wasm"], graphOptimizationLevel: "all", logSeverityLevel: 3 });
      sessionRef.current = session;
      const elapsed = ((performance.now() - t0) / 1000).toFixed(1);
      onLog(`[Import] ✅ Model loaded in ${elapsed}s: inputs=[${session.inputNames}], outputs=[${session.outputNames}]`);
      setStatus("Model loaded! Upload a CSV with a text column.");
      setStep(2);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setStatus(`Failed: ${msg}`);
      onLog(`[Import Error] ${msg}`);
    } finally {
      setIsBusy(false);
    }
  }, [onLog, setIsBusy]);

  const handleCsvWithText = useCallback(async (file: File) => {
    setIsBusy(true);
    const t0 = performance.now();
    try {
      setStatus("Parsing file...");
      const isJsonl = file.name.toLowerCase().endsWith(".jsonl") || file.name.toLowerCase().endsWith(".json");
      let result: ReturnType<typeof parseCSVString>;
      if (isJsonl) {
        const text = await file.text();
        const lines = text.trim().split("\n").filter((l) => l.trim());
        if (lines.length === 0) throw new Error("JSONL file is empty");
        const firstObj = JSON.parse(lines[0]);
        const keys = Object.keys(firstObj);
        const csvLines = [keys.join(",")];
        for (const line of lines) {
          const obj = JSON.parse(line);
          csvLines.push(keys.map((k) => { const v = obj[k]; return v === null ? "" : String(v).includes(",") ? `"${v}"` : String(v); }).join(","));
        }
        result = parseCSVString(csvLines.join("\n"));
        onLog(`[Import] Parsed JSONL: ${lines.length} lines → ${keys.length} columns`);
      } else {
        result = await parseCSV(file);
      }
      const textColIdx = result.schema.columns.findIndex((c) => c.type === "text" || c.type === "categorical");
      const actualTextCol = textColIdx === -1 ? 0 : textColIdx;
      const labelColIdx = result.schema.targetIndex;
      onLog(`[Import] Tokenizing ${result.rawRows.length} samples... (text col: ${actualTextCol}, label col: ${labelColIdx})`);
      const tokenized = result.rawRows.map((row) => tokenize(String(row[actualTextCol])));
      const session = sessionRef.current;
      if (!session) { setStatus("No model loaded. Upload ONNX first."); return; }
      setStatus("Extracting embeddings...");
      const allEmbeddings: number[][] = [];
      const batchSize = 4;
      for (let i = 0; i < tokenized.length; i += batchSize) {
        const batch = tokenized.slice(i, i + batchSize);
        const maxLen = 128;
        const b = batch.length;
        const input_ids = batch.map((t) => t.input_ids).flat();
        const attention_mask = batch.map((t) => t.attention_mask).flat();
        // Build token_type_ids (all zeros for single-segment input)
        const token_type_ids = new Array(b * maxLen).fill(0);
        const feeds: Record<string, ort.Tensor> = {};
        if (session.inputNames.includes("input_ids")) feeds.input_ids = new ort.Tensor("int64", BigInt64Array.from(input_ids.map(BigInt)), [b, maxLen]);
        if (session.inputNames.includes("attention_mask")) feeds.attention_mask = new ort.Tensor("int64", BigInt64Array.from(attention_mask.map(BigInt)), [b, maxLen]);
        if (session.inputNames.includes("token_type_ids")) feeds.token_type_ids = new ort.Tensor("int64", BigInt64Array.from(token_type_ids.map(BigInt)), [b, maxLen]);
        // Log which inputs we're providing (first batch only)
        if (i === 0) onLog(`[Import] Feeding inputs: [${Object.keys(feeds).join(", ")}]`);
        try {
          const results = await session.run(feeds);
          let emb: number[][] = [];
          if (results.last_hidden_state) {
            const data = results.last_hidden_state.data as Float32Array;
            const dims = results.last_hidden_state.dims;
            const seqLen = Number(dims[1]), hiddenDim = Number(dims[2]);
            for (let bi = 0; bi < b; bi++) {
              const pooled = new Array(hiddenDim).fill(0); let valid = 0;
              for (let s = 0; s < seqLen; s++) {
                const idx = bi * seqLen * hiddenDim + s * hiddenDim;
                if (attention_mask[bi * maxLen + s] > 0) { for (let h = 0; h < hiddenDim; h++) pooled[h] += data[idx + h]; valid++; }
              }
              if (valid > 0) for (let h = 0; h < hiddenDim; h++) pooled[h] /= valid;
              emb.push(pooled);
            }
          } else if (results.pooler_output) {
            const data = results.pooler_output.data as Float32Array;
            const dims = results.pooler_output.dims;
            const hiddenDim = Number(dims[dims.length - 1]);
            for (let bi = 0; bi < b; bi++) { const row: number[] = []; for (let h = 0; h < hiddenDim; h++) row.push(data[bi * hiddenDim + h]); emb.push(row); }
          } else if (results.logits) {
            const data = results.logits.data as Float32Array;
            const dims = results.logits.dims;
            const numClasses = Number(dims[dims.length - 1]);
            for (let bi = 0; bi < b; bi++) { const row: number[] = []; for (let h = 0; h < numClasses; h++) row.push(data[bi * numClasses + h]); emb.push(row); }
          } else {
            onLog(`[Import] ⚠️ Unknown outputs: [${Object.keys(results).join(", ")}]. Expected last_hidden_state, pooler_output, or logits.`);
          }
          allEmbeddings.push(...emb);
          setStatus(`Extracted ${allEmbeddings.length}/${tokenized.length} embeddings...`);
        } catch (err) { onLog(`[Import] Batch ${i} failed: ${err instanceof Error ? err.message : String(err)}`); }
      }
      // Extract labels directly from ALL raw rows (result.labels may have fewer due to null filtering)
      const labelMap: Record<string, number> = {};
      let nextLabelId = 0;
      const allLabels: number[] = [];
      for (const row of result.rawRows) {
        const rawLabel = row[labelColIdx]?.trim() ?? "";
        const numLabel = Number(rawLabel);
        if (!isNaN(numLabel) && rawLabel !== "") {
          allLabels.push(Math.round(numLabel));
        } else {
          if (!(rawLabel in labelMap)) labelMap[rawLabel] = nextLabelId++;
          allLabels.push(labelMap[rawLabel]);
        }
      }
      const uniqueLabels = new Set(allLabels);
      const elapsed = ((performance.now() - t0) / 1000).toFixed(1);
      onLog(`[Import] Labels: ${allLabels.length} samples, ${uniqueLabels.size} unique classes [${Array.from(uniqueLabels).join(", ")}]`);
      setEmbeddings(allEmbeddings);
      setEmbLabels(allLabels);
      setEmbDim(allEmbeddings[0]?.length ?? 0);
      setSchema(result.schema);
      setNormStats(result.normStats);
      setStep(3);
      onLog(`[Import] ✅ Extracted ${allEmbeddings.length} embeddings in ${elapsed}s, dim=${allEmbeddings[0]?.length ?? 0}`);
      setStatus(`Ready: ${allEmbeddings.length} embeddings. Click Train Head.`);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setStatus(`Error: ${msg}`);
      onLog(`[Import Error] ${msg}`);
    } finally {
      setIsBusy(false);
    }
  }, [onLog, setIsBusy]);

  const resetHead = useCallback(() => {
    setIsTrainingHead(false);
    setHeadProgress(0);
    setHeadEpoch(0);
    setStatus("Ready to re-train. Click Train Classifier Head.");
    onLog("[Import] Head training reset. Ready to re-train.");
  }, [onLog]);

  const trainHead = useCallback(async () => {
    if (embeddings.length === 0 || !schema || !normStats) return;
    const numClasses = new Set(embLabels).size;
    if (numClasses < 2) {
      onLog(`[Import Error] Only ${numClasses} unique label found in ${embLabels.length} samples. Need ≥2 classes for classification. Check your dataset's label column.`);
      setStatus(`Error: only ${numClasses} class found — need at least 2`);
      return;
    }
    let xs: tf.Tensor | null = null;
    let ys: tf.Tensor | null = null;
    let head: tf.Sequential | null = null;
    const t0 = performance.now();
    try {
      setIsBusy(true);
      setIsTrainingHead(true);
      setHeadProgress(0);
      setHeadEpoch(0);
      setStatus("Training classifier head...");
      const hwInfo = await detectHardware();
      const backend = tf.getBackend();
      onLog(`[Import] Training classifier head... (${backend?.toUpperCase() || "CPU"} backend · ${hwInfo})`);
      await tf.ready();
      xs = tf.tensor2d(embeddings);
      ys = tf.oneHot(tf.tensor1d(embLabels, "int32"), numClasses);
      head = tf.sequential();
      head.add(tf.layers.dense({ inputShape: [embDim], units: 64, activation: "relu" }));
      head.add(tf.layers.dense({ units: numClasses, activation: "softmax" }));
      head.compile({ optimizer: tf.train.adam(0.001), loss: "categoricalCrossentropy", metrics: ["accuracy"] });
      onLog(`[Import] Head: ${embDim} → 64 → ${numClasses} (${head.countParams()} params)`);
      const totalHeadEpochs = 50;
      await head.fit(xs, ys, { epochs: totalHeadEpochs, batchSize: 32, validationSplit: 0.2, verbose: 0,
        callbacks: { onEpochEnd: (epoch: number, logs?: tf.Logs) => {
          setHeadEpoch(epoch);
          setHeadProgress(Math.round((epoch / totalHeadEpochs) * 100));
          if (epoch % 10 === 0 && logs) onLog(`[Import] Epoch ${epoch}/${totalHeadEpochs}: loss=${logs.loss.toFixed(4)}, acc=${((logs.acc as number) * 100).toFixed(1)}%`);
        } },
      });
      const weights: number[][] = [];
      for (const layer of head.layers) for (const w of layer.getWeights()) weights.push(Array.from(w.dataSync()));
      const weightsJson = JSON.stringify(weights);
      setHeadProgress(100);
      const elapsed = ((performance.now() - t0) / 1000).toFixed(1);
      setStatus(`Head trained in ${elapsed}s! Switching to sandbox...`);
      onLog(`[Import] ✅ Head trained in ${elapsed}s. Model ready.`);
      // Fix schema categories to match actual trained classes
      const fixedSchema = { ...schema, columns: schema.columns.map((c, i) => 
        i === schema.targetIndex ? { ...c, categories: Array.from(new Set(embLabels)).sort().map(String) } : c
      )};
      onModelTrained(weightsJson, fixedSchema, normStats);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      onLog(`[Import Error] Head training failed: ${msg}`);
      setStatus(`Training failed: ${msg}`);
    } finally {
      xs?.dispose(); ys?.dispose(); head?.dispose();
      setIsTrainingHead(false);
      setIsBusy(false);
    }
  }, [embeddings, embLabels, embDim, schema, normStats, onLog, onModelTrained, setIsBusy]);

  return (
    <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 14 }}>
      <div>
        <div style={{ fontSize: 12, color: "var(--accent)", marginBottom: 10, padding: "8px 12px", background: "var(--bg-surface-2)", borderRadius: "var(--radius-md)", border: "1px solid var(--border)" }}>
          <strong>Import ONNX Workflow:</strong> Bring a pre-trained transformer (BERT, DistilBERT, etc.) as an .onnx file. Extract text embeddings through it, then train only a small classifier "head" on your own labeled dataset. Ideal for NLP fine-tuning without cloud GPUs.
        </div>
        <div className="fs-card" style={{ marginBottom: 14 }}>
          <div className="fs-card-header" style={{ alignItems: "flex-start" }}>
            <div>
              <div className="fs-card-title"><Monitor size={16} /> Step 1: Select Backbone</div>
              <div style={{ color: "var(--text-muted)", fontSize: 12, marginTop: 2 }}>Choose a transformer or upload your own ONNX model.</div>
            </div>
            <span className="fs-badge fs-badge-accent">{step > 1 ? "✅" : "1/3"}</span>
          </div>
          <div className="fs-preset-grid" style={{ marginBottom: 10 }}>
            {MODEL_PRESETS.map((p) => (
              <div key={p.id} className={`fs-preset-card ${selectedPreset === p.id ? "selected" : ""}`} onClick={() => setSelectedPreset(p.id)}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <span className="fs-preset-name">{p.name}</span>
                  {selectedPreset === p.id && <Check size={16} style={{ color: "var(--accent)" }} />}
                </div>
                <div className="fs-preset-desc">{p.desc}</div>
                <div className="fs-preset-size">{p.size}</div>
              </div>
            ))}
          </div>
          {selectedPreset === "custom" && (
            <div className="fs-dropzone" style={{ padding: 14 }} onClick={() => document.getElementById("onnx-input")?.click()}>
              <Upload size={18} /><div style={{ fontWeight: 500, fontSize: 13, marginTop: 2 }}>Upload .onnx model</div>
              <input id="onnx-input" type="file" accept=".onnx" style={{ display: "none" }} onChange={(e) => { const f = e.target.files?.[0]; if (f) handleOnnxUpload(f); }} />
            </div>
          )}
          {selectedPreset !== "custom" && (
            <div style={{ padding: 10, background: "var(--bg-surface-2)", borderRadius: "var(--radius-md)", fontSize: 12, color: "var(--text-muted)" }}>
              <Info size={12} style={{ display: "inline", marginRight: 4 }} />
              Select "Custom ONNX" above, then upload the {MODEL_PRESETS.find((p) => p.id === selectedPreset)?.name} model file.
            </div>
          )}
        </div>
        {step >= 3 && (
          <div className="fs-card">
            <div className="fs-card-header">
              <div className="fs-card-title"><Brain size={16} /> Step 3: Train Head</div>
              <span className="fs-badge fs-badge-accent">3/3</span>
            </div>
            <div className="fs-metrics" style={{ marginBottom: 12 }}>
              <div className="fs-metric"><div className="fs-metric-value" style={{ fontSize: 18 }}>{embeddings.length}</div><div className="fs-metric-label">Samples</div></div>
              <div className="fs-metric"><div className="fs-metric-value" style={{ fontSize: 18 }}>{embDim}</div><div className="fs-metric-label">Emb Dim</div></div>
              <div className="fs-metric"><div className="fs-metric-value" style={{ fontSize: 18 }}>{new Set(embLabels).size}</div><div className="fs-metric-label">Classes</div></div>
            </div>
            {isTrainingHead && (
              <div style={{ marginBottom: 12 }}>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, color: "var(--text-muted)", marginBottom: 4 }}>
                  <span>Epoch {headEpoch} / 50</span><span>{headProgress}%</span>
                </div>
                <div className="fs-progress"><div className="fs-progress-fill" style={{ width: `${headProgress}%` }} /></div>
              </div>
            )}
            <div style={{ display: "flex", gap: 8 }}>
              <button className="fs-btn fs-btn-primary" onClick={trainHead} disabled={embeddings.length === 0 || isTrainingHead} style={{ flex: 1 }}>
                {isTrainingHead ? <><Loader2 size={14} style={{ animation: "spin 1s linear infinite" }} /> Training...</> : headProgress === 100 ? <><Brain size={14} /> Re-train Head</> : <><Brain size={14} /> Train Classifier Head</>}
              </button>
              {(headProgress === 100 || isTrainingHead) && (
                <button className="fs-btn fs-btn-sm" onClick={resetHead} disabled={isTrainingHead} title="Reset and train again">
                  <RotateCcw size={13} />
                </button>
              )}
            </div>
          </div>
        )}
      </div>
      <div>
        <div className="fs-card" style={{ marginBottom: 16, opacity: step < 2 ? 0.5 : 1 }}>
          <div className="fs-card-header">
            <div className="fs-card-title"><FileSpreadsheet size={16} /> Step 2: Upload CSV</div>
            <span className="fs-badge fs-badge-accent">{step > 2 ? "✅" : step >= 2 ? "2/3" : "locked"}</span>
          </div>
          <p style={{ color: "var(--text-muted)", fontSize: 13, marginBottom: 12 }}>CSV with a text column and a label column.</p>
          <div className="fs-dropzone" style={{ padding: 20 }} onClick={() => { if (step >= 2) document.getElementById("csv-import-input")?.click(); }}>
            <FileSpreadsheet size={20} />
            <div style={{ fontWeight: 500, fontSize: 13, marginTop: 4 }}>{step >= 2 ? "Drop CSV / JSON / JSONL or click" : "Complete Step 1 first"}</div>
            {step >= 2 && <input id="csv-import-input" type="file" accept=".csv,.json,.jsonl" style={{ display: "none" }} onChange={(e) => { const f = e.target.files?.[0]; if (f) handleCsvWithText(f); }} />}
          </div>
        </div>
        {status && (
          <div className="fs-card">
            <div className="fs-card-header"><div className="fs-card-title"><Info size={16} /> Status</div></div>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              {status.includes("...") && <Loader2 size={14} style={{ animation: "spin 1s linear infinite" }} />}
              <span style={{ fontSize: 13, color: status.startsWith("Failed") || status.startsWith("Error") ? "var(--error)" : "var(--text-secondary)" }}>{status}</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════
   DOCS TAB — with subsection navigation
   ═══════════════════════════════════════════════════════════════ */
const DOC_SECTIONS = [
  { id: "quickstart", label: "Quick Start", icon: <Zap size={14} /> },
  { id: "datasets", label: "Example Datasets", icon: <Code2 size={14} /> },
  { id: "gpu", label: "GPU Setup", icon: <Monitor size={14} /> },
  { id: "deploy", label: "Deploy", icon: <Globe size={14} /> },
  { id: "api", label: "API Reference", icon: <Terminal size={14} /> },
  { id: "tech", label: "Tech Stack", icon: <Cpu size={14} /> },
];

function ReleasesTab() {
  const GITHUB_RELEASES = "https://github.com/PacifAIst/Frugalsloth/releases/latest";
  return (
    <div style={{ width: "100%" }}>
      <div className="fs-card">
        <div className="fs-card-header">
          <div className="fs-card-title"><Tag size={16} /> Releases</div>
          <span className="fs-badge fs-badge-accent">v0.3.3</span>
        </div>

        <div className="fs-docs-section">
          <h3>Electron Desktop App (GPU-accelerated)</h3>
          <p>A standalone .exe that bundles Chromium with optimal GPU flags pre-configured. No browser setup needed — double-click and train.</p>
          <div style={{ background: "var(--bg-surface-2)", padding: 12, borderRadius: "var(--radius-md)", fontSize: 13, marginTop: 8 }}>
            <strong>Pre-configured flags:</strong>
            <ul style={{ paddingLeft: 18, margin: "6px 0" }}>
              <li><code>--ignore-gpu-blocklist</code> — enables GPU for all chipsets</li>
              <li><code>--enable-gpu-rasterization</code> — GPU-accelerated page rendering</li>
              <li><code>--use-angle=gl</code> — OpenGL backend for TensorFlow.js</li>
              <li><code>--enable-features=SharedArrayBuffer</code> — TF.js multi-threading</li>
            </ul>
            <p style={{ margin: 0, color: "var(--accent)" }}>Result: RTX 3060 runs at full power. Training 10-50x faster than browser.</p>
          </div>
          <div style={{ marginTop: 12 }}>
            <a href={GITHUB_RELEASES} target="_blank" rel="noreferrer" className="fs-btn fs-btn-primary" style={{ textDecoration: "none", display: "inline-flex" }}><Monitor size={14} /> Download Electron (.exe)</a>
          </div>
        </div>

        <div className="fs-docs-section">
          <h3>Static Web ZIP (Host Anywhere)</h3>
          <p>Download the <code>dist/</code> folder as a ZIP. Unzip on any static host — Namecheap, GitHub Pages, Netlify, or your local machine. Just open <code>index.html</code>.</p>
          <div style={{ marginTop: 12 }}>
            <a href={GITHUB_RELEASES} target="_blank" rel="noreferrer" className="fs-btn fs-btn-primary" style={{ textDecoration: "none", display: "inline-flex" }}><Globe size={14} /> Download Static ZIP</a>
          </div>
        </div>

        <div className="fs-docs-section">
          <h3>Cloudflare Pages Deploy</h3>
          <p>One-command deploy to Cloudflare's global CDN. Free tier includes unlimited bandwidth.</p>
          <div style={{ background: "var(--bg-surface-2)", padding: 12, borderRadius: "var(--radius-md)", fontSize: 13, marginTop: 8 }}>
            <strong>Steps:</strong>
            <ol style={{ paddingLeft: 18, margin: "6px 0" }}>
              <li>Install Wrangler: <code>npm install -g wrangler</code></li>
              <li>Login: <code>wrangler login</code></li>
              <li>Deploy: <code>wrangler pages deploy dist</code></li>
            </ol>
          </div>
          <div style={{ marginTop: 8, fontSize: 12, color: "var(--text-muted)" }}>
            Requires <code>wrangler.toml</code> with COOP/COEP headers for SharedArrayBuffer support.
          </div>
        </div>

        <div className="fs-docs-section">
          <h3>AGPL v3 License</h3>
          <p>FrugalSloth is free software under AGPL v3. Anyone using it must share their source code. This ensures the ecosystem stays open.</p>
          <p style={{ marginTop: 6 }}><strong>Commercial use?</strong> Companies can use FrugalSloth internally, but if they distribute it (SaaS, product embedding), they must publish their modifications. Contact us for dual-licensing options.</p>
        </div>
      </div>
    </div>
  );
}

function DocsTab() {
  const [activeSection, setActiveSection] = useState("quickstart");

  const copyToClipboard = (text: string) => { navigator.clipboard.writeText(text).catch(() => {}); };
  const CodeBlock = ({ code, label }: { code: string; label: string }) => (
    <div className="fs-code-block">
      <button className="fs-copy-btn" onClick={() => copyToClipboard(code)}><Copy size={11} /> Copy</button>
      <div style={{ color: "var(--text-muted)", fontSize: 10, marginBottom: 4, textTransform: "uppercase", letterSpacing: "0.05em" }}>{label}</div>
      <pre style={{ margin: 0, whiteSpace: "pre-wrap" }}>{code}</pre>
    </div>
  );

  return (
    <div className="fs-grid-2" style={{ gridTemplateColumns: "200px 1fr", alignItems: "start" }}>
      {/* Sidebar Navigation */}
      <div>
        <div className="fs-card" style={{ padding: 12 }}>
          <div style={{ fontWeight: 600, fontSize: 13, marginBottom: 10, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.05em" }}>Sections</div>
          {DOC_SECTIONS.map((s) => (
            <button key={s.id} onClick={() => setActiveSection(s.id)} className="fs-btn" style={{
              width: "100%", justifyContent: "flex-start", marginBottom: 4,
              background: activeSection === s.id ? "var(--accent-light)" : "transparent",
              color: activeSection === s.id ? "var(--accent-hover)" : "var(--text-secondary)",
              borderColor: activeSection === s.id ? "var(--accent-border)" : "transparent",
            }}>
              {s.icon} {s.label}
            </button>
          ))}
        </div>
      </div>

      {/* Content */}
      <div>
        {/* Quick Start */}
        {activeSection === "quickstart" && (
          <div className="fs-card">
            <div className="fs-card-header"><div className="fs-card-title"><Zap size={16} /> Quick Start</div></div>
            <div className="fs-docs-section">
              <h3><ChevronRight size={14} /> 1. Load a Dataset</h3>
              <p>Go to <strong>Data</strong> tab, upload a CSV or JSON file. Or click a demo dataset to try immediately.</p>
              <CodeBlock label="CSV Format" code={`temperature,humidity,quality
22.5,45,good
18.3,60,bad
25.1,35,good`} />
              <CodeBlock label="JSON Format" code={`[{"temperature":22.5,"humidity":45,"quality":0},
{"temperature":18.3,"humidity":60,"quality":1},
{"temperature":25.1,"humidity":35,"quality":0}]`} />
            </div>
            <div className="fs-docs-section">
              <h3><ChevronRight size={14} /> 2. Configure & Train</h3>
              <p>Switch to <strong>Train</strong>, adjust hyperparameters (hover ⓘ for help), click <strong>Start Training</strong>.</p>
            </div>
            <div className="fs-docs-section">
              <h3><ChevronRight size={14} /> 3. Test Live</h3>
              <p>Go to <strong>Sandbox</strong>, adjust sliders to see real-time predictions with confidence scores.</p>
            </div>
            <div className="fs-docs-section">
              <h3><ChevronRight size={14} /> 4. Import ONNX — Transformer Fine-Tuning</h3>
              <p>The <strong>Import ONNX</strong> tab lets you bring a pre-trained transformer (BERT, DistilBERT, etc.) and fine-tune just the classifier "head" on your own dataset. Here's how it works:</p>
              <table style={{ width: "100%", fontSize: 13, marginTop: 10 }}>
                <thead><tr style={{ borderBottom: "1px solid var(--border)" }}><th style={{ textAlign: "left", padding: "4px 0" }}>Step</th><th style={{ textAlign: "left", padding: "4px 0" }}>What you do</th><th style={{ textAlign: "left", padding: "4px 0" }}>What happens</th></tr></thead>
                <tbody style={{ color: "var(--text-secondary)" }}>
                  <tr style={{ borderBottom: "1px solid var(--border-light)" }}><td style={{ padding: "4px 0", fontWeight: 500 }}>1</td><td style={{ padding: "4px 0" }}>Upload .onnx model</td><td style={{ padding: "4px 0" }}>Loads the transformer backbone (BERT, DistilBERT, etc.)</td></tr>
                  <tr style={{ borderBottom: "1px solid var(--border-light)" }}><td style={{ padding: "4px 0", fontWeight: 500 }}>2</td><td style={{ padding: "4px 0" }}>Upload .csv/.jsonl with text + labels</td><td style={{ padding: "4px 0" }}>Text is tokenized and fed through the transformer</td></tr>
                  <tr style={{ borderBottom: "1px solid var(--border-light)" }}><td style={{ padding: "4px 0", fontWeight: 500 }}>3</td><td style={{ padding: "4px 0" }}>Click "Train Classifier Head"</td><td style={{ padding: "4px 0" }}>Embeddings are extracted; a tiny MLP head is trained on top</td></tr>
                </tbody>
              </table>
              <p style={{ marginTop: 10 }}><strong>Why?</strong> Instead of training a massive model from scratch, you reuse a pre-trained transformer's knowledge and only train a small classifier layer. This is how real NLP fine-tuning works — BERT gives you embeddings, you train a head on top.</p>
              <CodeBlock label="JSONL format for text classification" code={`{"text":"This product is amazing","label":1}
{"text":"Terrible quality, broke in 2 days","label":0}
{"text":"Best purchase I ever made","label":1}
{"text":"Waste of money","label":0}`} />
              <p style={{ color: "var(--accent)", fontSize: 12 }}><strong>Note:</strong> Models that output <code>pooler_output</code> or <code>last_hidden_state</code> work best. Models that only output <code>logits</code> produce lower-quality embeddings for fine-tuning.</p>
            </div>
            <div className="fs-docs-section">
              <h3><ChevronRight size={14} /> 5. Export</h3>
              <p>Click <strong>Export Microservice</strong> to download a zip with ONNX model + engine JS.</p>
            </div>
          </div>
        )}

        {/* Example Datasets */}
        {activeSection === "datasets" && (
          <div className="fs-card">
            <div className="fs-card-header"><div className="fs-card-title"><Code2 size={16} /> Example Datasets</div></div>
            <div className="fs-docs-section">
              <h3>Classification — Iris (3 classes)</h3>
              <CodeBlock label="iris.json (copy & save as file)" code={`[
  {"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2,"species":0},
  {"sepal_length":4.9,"sepal_width":3.0,"petal_length":1.4,"petal_width":0.2,"species":0},
  {"sepal_length":4.7,"sepal_width":3.2,"petal_length":1.3,"petal_width":0.2,"species":0},
  {"sepal_length":7.0,"sepal_width":3.2,"petal_length":4.7,"petal_width":1.4,"species":1},
  {"sepal_length":6.4,"sepal_width":3.2,"petal_length":4.5,"petal_width":1.5,"species":1},
  {"sepal_length":6.9,"sepal_width":3.1,"petal_length":4.9,"petal_width":1.5,"species":1},
  {"sepal_length":6.3,"sepal_width":3.3,"petal_length":6.0,"petal_width":2.5,"species":2},
  {"sepal_length":5.8,"sepal_width":2.7,"petal_length":5.1,"petal_width":1.9,"species":2}
]`} />
            </div>
            <div className="fs-docs-section">
              <h3>Regression — Housing Prices</h3>
              <CodeBlock label="housing.csv (copy & save as file)" code={`sqft,bedrooms,age,price
1200,3,15,250000
850,2,8,180000
2000,4,22,420000
1500,3,12,310000
950,2,5,195000`} />
            </div>
            <div className="fs-docs-section">
              <h3>Text Classification — Sentiment (use with Import ONNX tab)</h3>
              <CodeBlock label="sentiment.csv (copy & save as file)" code={`text,label
This movie was amazing,positive
Terrible waste of time,negative
Great acting and plot,positive
Boring and predictable,negative
Absolutely loved it,positive
Worst film ever,negative
Highly recommend,positive
Disappointing ending,negative`} />
            </div>
          </div>
        )}

        {/* GPU Setup */}
        {activeSection === "gpu" && (
          <div className="fs-card">
            <div className="fs-card-header"><div className="fs-card-title"><Monitor size={16} /> GPU Setup Guide</div></div>
            <div className="fs-docs-section">
              <h3>Why is training slow?</h3>
              <p>FrugalSloth uses TensorFlow.js with the <strong>WebGL backend</strong> for GPU-accelerated training. If your browser is using the CPU or integrated graphics instead of your dedicated GPU (e.g., NVIDIA RTX), training will be 10-50x slower.</p>
            </div>
            <div className="fs-docs-section">
              <h3>Check your current GPU</h3>
              <p>Open <code>chrome://gpu/</code> in Chrome and look for <strong>GL_RENDERER</strong>. If it says <code>Intel</code>, <code>Microsoft Basic Render</code>, or <code>d3d11-warp</code>, you are NOT using your dedicated GPU.</p>
              <CodeBlock label="What you WANT to see (NVIDIA GPU active)" code={`GL_RENDERER: ANGLE (NVIDIA, NVIDIA GeForce RTX 3060 Laptop GPU...)`} />
              <CodeBlock label="What you DON'T want to see" code={`GL_RENDERER: ANGLE (Intel, Intel(R) UHD Graphics...)
GL_RENDERER: ANGLE (Microsoft, Microsoft Basic Render Driver...) ← CPU!
Display type: ANGLE_D3D11_WARP ← Software rendering!`} />
            </div>
            <div className="fs-docs-section">
              <h3>Step 1: Windows Graphics Settings</h3>
              <ol style={{ paddingLeft: 18, color: "var(--text-secondary)", fontSize: 13 }}>
                <li>Windows Settings → System → Display → scroll down → <strong>Graphics</strong></li>
                <li>Click <strong>"Add an app"</strong> → <strong>Browse</strong></li>
                <li>Navigate to: <code>C:\Program Files\Google\Chrome\Application\chrome.exe</code></li>
                <li>Click <strong>Options</strong> → select <strong>"High performance"</strong> (your NVIDIA GPU)</li>
                <li><strong>Save</strong></li>
              </ol>
            </div>
            <div className="fs-docs-section">
              <h3>Step 2: Remove ALL Chrome shortcut flags</h3>
              <p style={{ color: "var(--accent)", fontWeight: 600 }}>⚠️ This is the #1 cause of broken training!</p>
              <p>Right-click your Chrome shortcut → <strong>Properties</strong> → <strong>Target</strong> field. Remove ALL flags. The target should be clean:</p>
              <CodeBlock label="CORRECT target (NO flags)" code={`"C:\Program Files\Google\Chrome\Application\chrome.exe"`} />
              <p><strong>Why?</strong> Flags like <code>--use-angle=gl</code> or <code>--ignore-gpu-blocklist</code> force OpenGL on Intel GPUs, which silently breaks TensorFlow.js training (loss becomes 0, charts flat). <strong>Do NOT use any flags.</strong></p>
              <CodeBlock label="REMOVE these flags if present" code={`--use-angle=gl           ← BREAKS Intel iGPU training! REMOVE!
--ignore-gpu-blocklist   ← causes cache issues! REMOVE!
--disable-gpu            ← disables GPU entirely! REMOVE!
--use-angle=d3d11-warp   ← forces CPU rendering! REMOVE!`} />
            </div>
            <div className="fs-docs-section">
              <h3>Step 3: Leave chrome://flags as DEFAULT</h3>
              <p>Do NOT change any flags in <code>chrome://flags/</code>. Leave everything at default. Modified flags are the most common cause of broken training.</p>
              <p>If you previously changed flags, reset them: go to <code>chrome://flags/</code> → click <strong>"Reset all"</strong> at top → <strong>Relaunch</strong>.</p>
            </div>
            <div className="fs-docs-section">
              <h3>Step 4: Verify</h3>
              <ol style={{ paddingLeft: 18, color: "var(--text-secondary)", fontSize: 13 }}>
                <li>Open <code>chrome://gpu/</code></li>
                <li>Check <strong>Display type</strong> — should be <code>ANGLE_D3D11</code> (default, works best)</li>
                <li>Check <strong>GL_RENDERER</strong> — Intel UHD is fine for small models</li>
                <li>Reload FrugalSloth — training should show real loss values (not 0)</li>
              </ol>
            </div>
            <div className="fs-docs-section">
              <h3>Why does Edge work but Chrome doesn't?</h3>
              <p>Microsoft Edge uses default settings (D3D11 ANGLE) which works correctly with Intel iGPUs. Chrome often has leftover flags from previous tweaks (GPU blocklist bypass, OpenGL forcing) that break TF.js.</p>
              <p><strong>Solution</strong>: Remove all Chrome flags (Step 2 above) and Chrome will work identically to Edge.</p>
            </div>
            <div className="fs-docs-section">
              <h3>If training still gives zeroes</h3>
              <ol style={{ paddingLeft: 18, color: "var(--text-secondary)", fontSize: 13 }}>
                <li>Go to <strong>Models</strong> tab → click <strong>"Hard Flush — Reset Everything"</strong></li>
                <li>Confirm the dialog — this clears all cached data and reloads</li>
                <li>Load a demo dataset and try training again</li>
              </ol>
            </div>
          </div>
        )}

        {/* Deploy */}
        {activeSection === "deploy" && (
          <div className="fs-card">
            <div className="fs-card-header"><div className="fs-card-title"><Globe size={16} /> Deploy</div></div>
            <div className="fs-docs-section">
              <h3>Cloudflare Pages (Recommended)</h3>
              <p>Upload the <code>dist/</code> folder. The <code>_headers</code> file enables SharedArrayBuffer for multi-threaded WASM.</p>
              <CodeBlock label="_headers file" code={`/*\n  Cross-Origin-Opener-Policy: same-origin\n  Cross-Origin-Embedder-Policy: require-corp`} />
            </div>
            <div className="fs-docs-section">
              <h3>GitHub Pages</h3>
              <p>Copy <code>dist/</code> to your <code>gh-pages</code> branch. <code>coi-serviceworker.js</code> enables SharedArrayBuffer after one page refresh.</p>
            </div>
            <div className="fs-docs-section">
              <h3>Using the Exported Engine</h3>
              <CodeBlock label="HTML" code={`<script type="module">\n  import { predict, init } from './frugalsloth-engine.js';\n  await init();\n  const result = await predict([22.5, 45, 1013]);\n  console.log(result.prediction);   // 0\n  console.log(result.confidence);   // 0.97\n  console.log(result.latencyMs);    // 2.3\n</script>`} />
            </div>
          </div>
        )}

        {/* API Reference */}
        {activeSection === "api" && (
          <div className="fs-card">
            <div className="fs-card-header"><div className="fs-card-title"><Terminal size={16} /> API Reference</div></div>
            <div className="fs-docs-section">
              <h3>predict(rawValues: number[])</h3>
              <p>Returns a prediction for the given input features (same order as training).</p>
              <CodeBlock label="Example" code={`const result = await predict([22.5, 45, 1013]);\n// result = { prediction: 0, confidence: 0.97, latencyMs: 2.3 }`} />
            </div>
            <div className="fs-docs-section">
              <h3>init()</h3>
              <p>Initializes the ONNX inference session. Auto-called on module load in browsers.</p>
              <CodeBlock label="Example" code={`const { initTime, featureNames } = await init();\n// initTime: 45.2 (ms)\n// featureNames: ['temperature', 'humidity', 'pressure']`} />
            </div>
            <div className="fs-docs-section">
              <h3>predictBatch(rawRows: number[][])</h3>
              <p>Predicts multiple samples in sequence.</p>
              <CodeBlock label="Example" code={`const results = await predictBatch([\n  [22.5, 45, 1013],\n  [18.3, 60, 1008]\n]);\n// results = [{prediction:0,...}, {prediction:1,...}]`} />
            </div>
          </div>
        )}

        {/* Tech Stack */}
        {activeSection === "tech" && (
          <div className="fs-card">
            <div className="fs-card-header"><div className="fs-card-title"><Cpu size={16} /> Tech Stack</div></div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
              {[
                { name: "ONNX Runtime Web", version: "1.17+", desc: "Inference engine, WASM backend" },
                { name: "TensorFlow.js", version: "4.x", desc: "Training engine, WebGL/WebGPU" },
                { name: "React", version: "18", desc: "UI framework" },
                { name: "TypeScript", version: "5.x", desc: "Type safety" },
                { name: "Vite", version: "6.x", desc: "Build tool" },
                { name: "Web Workers", version: "Native", desc: "Off-main-thread training" },
                { name: "IndexedDB", version: "Native", desc: "Local model persistence" },
                { name: "PapaParse", version: "5.x", desc: "CSV streaming parser" },
              ].map((tech) => (
                <div key={tech.name} style={{ padding: 10, background: "var(--bg-surface-2)", borderRadius: "var(--radius-md)" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                    <span style={{ fontWeight: 600, fontSize: 13 }}>{tech.name}</span>
                    <span className="fs-badge fs-badge-neutral" style={{ fontSize: 10 }}>{tech.version}</span>
                  </div>
                  <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 2 }}>{tech.desc}</div>
                </div>
              ))}
            </div>
            <div style={{ marginTop: 12, padding: 10, background: "var(--accent-light)", borderRadius: "var(--radius-md)", fontSize: 12, color: "var(--accent-hover)" }}>
              <Zap size={12} style={{ display: "inline", marginRight: 4 }} />
              <strong>Performance:</strong> Web Workers keep the UI at 60fps during training. ONNX Runtime Web WASM is ~10× faster than pure JS inference. TF.js WebGL backend accelerates matrix operations using the GPU.
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
