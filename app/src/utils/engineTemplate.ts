/**
 * ============================================================================
 * FrugalSloth — Engine Template Generator
 * ============================================================================
 * Generates the frugalsloth-engine.js ES module that wraps the ONNX model
 * with preprocessing and exposes a universal predict() API.
 *
 * The generated engine is a self-contained microservice that:
 *   - Loads the embedded (base64) or sibling ONNX model
 *   - Handles input normalization (z-score, label encoding)
 *   - Runs inference via ONNX Runtime Web
 *   - Returns prediction + confidence + latency
 * ============================================================================
 */

import type { DatasetSchema, NormalizationStats } from "@/types/frugalsloth";

/** Generate the frugalsloth-engine.js source */
export function generateEngine(
  modelOnnx: Uint8Array,
  schema: DatasetSchema,
  normStats: NormalizationStats,
  modelName: string,
  embedModel: boolean
): { engineJs: string; readme: string } {
  // Build normalization code
  const numericNorms = Object.entries(normStats.numeric).map(([colIdx, stats]) => ({
    featureIdx: schema.columns.filter((_, i) => i < schema.targetIndex).filter((_, i) => i < parseInt(colIdx)).length +
                (parseInt(colIdx) > schema.targetIndex ? -1 : 0),
    mean: stats.mean,
    std: stats.std,
  }));

  const catEncodings = Object.entries(normStats.categorical).map(([colIdx, mapping]) => ({
    featureIdx: schema.columns.filter((_, i) => i < schema.targetIndex).filter((_, i) => i < parseInt(colIdx)).length +
                (parseInt(colIdx) > schema.targetIndex ? -1 : 0),
    mapping,
  }));

  const featureNames = schema.columns
    .filter((_, i) => i !== schema.targetIndex)
    .map((c) => c.name);

  const targetName = schema.columns[schema.targetIndex].name;
  const isClassification = schema.taskType !== "regression";
  const isBinary = schema.taskType === "binary_classification";
  const categories = schema.columns[schema.targetIndex].categories;

  // Base64 encode the model if embedding
  const modelBase64 = embedModel
    ? btoa(String.fromCharCode(...modelOnnx))
    : null;

  const engineJs = `/**
 * ============================================================================
 * FrugalSloth Engine — Self-Contained Edge AI Microservice
 * ============================================================================
 * Model: ${modelName}
 * Task: ${schema.taskType}
 * Features: ${featureNames.join(", ")}
 * Target: ${targetName}
 * Size: ${(modelOnnx.length / 1024).toFixed(1)}KB
 *
 * Usage:
 *   import { predict, init } from './frugalsloth-engine.js';
 *   await init();
 *   const result = await predict([${featureNames.map(() => 0).join(", ")}]);
 *   console.log(result.prediction, result.confidence, result.latencyMs);
 * ============================================================================
 */

// Feature configuration
const FEATURE_NAMES = ${JSON.stringify(featureNames)};
const TARGET_NAME = "${targetName}";
const IS_CLASSIFICATION = ${isClassification};
const IS_BINARY = ${isBinary};
const CATEGORIES = ${JSON.stringify(categories)};

// Normalization statistics
const NUMERIC_NORMS = ${JSON.stringify(numericNorms)};
const CAT_ENCODINGS = ${JSON.stringify(catEncodings)};

// Model data
const MODEL_B64 = "${modelBase64 ?? ""}";

let session = null;
let initTime = 0;

/** Initialize the inference engine */
export async function init() {
  const t0 = performance.now();
  const ort = await import("onnxruntime-web");

  let modelBuffer;
  if (MODEL_B64) {
    // Load from embedded base64
    const binary = atob(MODEL_B64);
    modelBuffer = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
      modelBuffer[i] = binary.charCodeAt(i);
    }
  } else {
    // Load from sibling .onnx file
    const resp = await fetch("./model.onnx");
    modelBuffer = new Uint8Array(await resp.arrayBuffer());
  }

  session = await ort.InferenceSession.create(modelBuffer, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });

  initTime = performance.now() - t0;

  // Self-test
  const testResult = await predictInternal(new Array(FEATURE_NAMES.length).fill(0));
  console.log(\`[FrugalSloth] Engine initialized in \${initTime.toFixed(1)}ms. Self-test: \${testResult.latencyMs.toFixed(1)}ms\`);

  return { initTime, featureNames: FEATURE_NAMES, targetName: TARGET_NAME };
}

/** Preprocess raw input values */
function preprocess(rawValues) {
  const features = [];

  for (let i = 0; i < rawValues.length; i++) {
    let val = rawValues[i];

    // Apply numeric normalization
    const norm = NUMERIC_NORMS.find(n => n.featureIdx === i);
    if (norm && norm.std > 0) {
      val = (val - norm.mean) / norm.std;
    }

    // Apply categorical encoding
    const cat = CAT_ENCODINGS.find(c => c.featureIdx === i);
    if (cat) {
      val = cat.mapping[String(val)] ?? 0;
    }

    features.push(val);
  }

  return features;
}

/** Internal inference (after preprocessing) */
async function predictInternal(processedFeatures) {
  if (!session) throw new Error("Engine not initialized. Call init() first.");

  const t0 = performance.now();
  const input = new Float32Array(processedFeatures);
  const tensor = new (await import("onnxruntime-web")).Tensor("float32", input, [1, processedFeatures.length]);

  const results = await session.run({ input: tensor });
  const output = results.output;
  const rawOutput = Array.from(output.data);

  const latencyMs = performance.now() - t0;

  // Post-process
  let prediction;
  let confidence;

  if (IS_CLASSIFICATION) {
    if (IS_BINARY) {
      const prob = rawOutput[0];
      prediction = prob > 0.5 ? 1 : 0;
      confidence = prob > 0.5 ? prob : 1 - prob;
    } else {
      const probs = rawOutput.map(v => Math.exp(v));
      const sum = probs.reduce((a, b) => a + b, 0);
      const normalized = probs.map(p => p / sum);
      const maxIdx = normalized.indexOf(Math.max(...normalized));
      prediction = CATEGORIES ? CATEGORIES[maxIdx] : maxIdx;
      confidence = normalized[maxIdx];
    }
  } else {
    prediction = rawOutput[0];
    confidence = 1.0;
  }

  return {
    prediction,
    confidence,
    latencyMs,
    rawOutput,
    processedFeatures,
  };
}

/**
 * Predict — the universal API.
 * @param {number[]} rawValues — raw feature values (same order as FEATURE_NAMES)
 * @returns {Promise<{prediction, confidence, latencyMs, rawOutput}>}
 */
export async function predict(rawValues) {
  const processed = preprocess(rawValues);
  return predictInternal(processed);
}

/**
 * Batch predict.
 * @param {number[][]} rawRows — array of raw feature arrays
 * @returns {Promise<Array>}
 */
export async function predictBatch(rawRows) {
  const results = [];
  for (const row of rawRows) {
    results.push(await predict(row));
  }
  return results;
}

// Auto-init if imported in browser
if (typeof window !== "undefined") {
  init().catch(() => {});
}
`;

  const readme = `# FrugalSloth Exported Model

**Model:** ${modelName}  
**Task:** ${schema.taskType}  
**Target:** ${targetName}

## Quick Start

\`\`\`html
<script type="module">
  import { predict, init } from './frugalsloth-engine.js';

  await init();

  // Predict with raw feature values
  const result = await predict([${featureNames.map(() => 0).join(", ")}]);
  console.log(result.prediction);
  console.log(result.confidence);
  console.log(result.latencyMs);
</script>
\`\`\`

## Features

| # | Name | Type |
|---|------|------|
${featureNames.map((n, i) => `| ${i} | ${n} | ${schema.columns.filter((_, j) => j !== schema.targetIndex)[i]?.type ?? "numeric"} |`).join("\n")}

## Model Info

- Parameters: ${modelOnnx.length} bytes (${(modelOnnx.length / 1024).toFixed(1)} KB)
- Engine: ONNX Runtime Web (WASM)
- Quantization: ${embedModel ? "Embedded (base64)" : "External (.onnx file)"}

## Deploy

Drop these files into any static hosting:
- Cloudflare Pages
- GitHub Pages
- Vercel / Netlify
- Raspberry Pi
- Cloudflare Worker (with bundler)

No backend required. 100% edge inference.
`;

  return { engineJs, readme };
}
