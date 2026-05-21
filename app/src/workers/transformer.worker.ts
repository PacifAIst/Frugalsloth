/**
 * ============================================================================
 * FrugalSloth — Transformer Feature Extraction Worker
 * ============================================================================
 * Runs a pre-trained ONNX transformer model (e.g., TinyBERT, DistilBERT)
 * to extract fixed-size embeddings from text inputs.
 *
 * Uses ONNX Runtime Web for backbone inference (not TF.js — ORT is faster
 * for transformer inference and handles the complex ops natively).
 *
 * The extracted embeddings are then used to train a small TF.js classifier
 * head in the main training worker.
 * ============================================================================
 */

import * as ort from "onnxruntime-web";

// ---------------------------------------------------------------------------
// Worker State
// ---------------------------------------------------------------------------

let session: ort.InferenceSession | null = null;
let embeddingDim = 768; // Default for BERT-like models
let modelLoaded = false;
let modelName = "";

// ---------------------------------------------------------------------------
// Message Handler
// ---------------------------------------------------------------------------

self.onmessage = async (event: MessageEvent) => {
  const { type, payload } = event.data;

  try {
    switch (type) {
      case "LOAD_MODEL":
        await handleLoadModel(payload);
        break;
      case "EXTRACT":
        await handleExtract(payload);
        break;
      case "GET_INFO":
        postMessage({
          type: "INFO",
          payload: { loaded: modelLoaded, embeddingDim, modelName },
        });
        break;
      default:
        postMessage({ type: "ERROR", payload: { message: `Unknown: ${type}` } });
    }
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    postMessage({ type: "ERROR", payload: { message: msg } });
  }
};

// ---------------------------------------------------------------------------
// LOAD_MODEL — Load an ONNX model from ArrayBuffer
// ---------------------------------------------------------------------------

async function handleLoadModel(payload: {
  modelBuffer: ArrayBuffer;
  name: string;
}) {
  postMessage({ type: "STATUS", payload: { message: "Loading ONNX model..." } });

  // Dispose old session
  if (session) {
    await session.release();
    session = null;
  }

  try {
    ort.env.wasm.wasmPaths = "/";
    ort.env.wasm.numThreads = 1;
    session = await ort.InferenceSession.create(payload.modelBuffer, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
      logSeverityLevel: 3,
    });

    modelName = payload.name;
    modelLoaded = true;

    // Inspect outputs to determine embedding dimension
    const outputNames = session.outputNames;
    if (outputNames.length > 0) {
      // Try to infer embedding dim from the first output
      // Most transformer models output [batch, seq_len, hidden_dim]
      // We take the [CLS] token or mean-pool
      embeddingDim = 768; // Will be refined on first extract
    }

    postMessage({
      type: "LOADED",
      payload: {
        name: modelName,
        inputs: session.inputNames,
        outputs: outputNames,
        embeddingDim,
      },
    });
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    postMessage({ type: "ERROR", payload: { message: `Failed to load model: ${msg}` } });
  }
}

// ---------------------------------------------------------------------------
// EXTRACT — Run inference to get embeddings from text
// ---------------------------------------------------------------------------

async function handleExtract(payload: {
  /** Array of { input_ids: number[], attention_mask: number[] } */
  tokenizedTexts: Array<{
    input_ids: number[];
    attention_mask: number[];
  }>;
  /** Max sequence length */
  maxLen?: number;
}) {
  if (!session) {
    postMessage({ type: "ERROR", payload: { message: "No model loaded" } });
    return;
  }

  const { tokenizedTexts, maxLen = 128 } = payload;
  const batchSize = tokenizedTexts.length;

  try {
    const inputNames = session.inputNames;

    // Determine input format
    const needsInputIds = inputNames.includes("input_ids");
    const needsAttentionMask = inputNames.includes("attention_mask");
    const needsTokenTypeIds = inputNames.includes("token_type_ids");

    // Pad/truncate all sequences to maxLen
    const paddedInputIds: number[][] = [];
    const paddedAttentionMask: number[][] = [];
    const paddedTokenTypeIds: number[][] = [];

    for (const tt of tokenizedTexts) {
      const ids = tt.input_ids.slice(0, maxLen);
      const mask = tt.attention_mask.slice(0, maxLen);
      // Pad
      while (ids.length < maxLen) { ids.push(0); mask.push(0); }
      paddedInputIds.push(ids);
      paddedAttentionMask.push(mask);
      paddedTokenTypeIds.push(new Array(maxLen).fill(0));
    }

    // Build feeds
    const feeds: Record<string, ort.Tensor> = {};

    if (needsInputIds) {
      const flatIds = paddedInputIds.flat();
      feeds.input_ids = new ort.Tensor("int64", BigInt64Array.from(flatIds.map(BigInt)), [batchSize, maxLen]);
    }
    if (needsAttentionMask) {
      const flatMask = paddedAttentionMask.flat();
      feeds.attention_mask = new ort.Tensor("int64", BigInt64Array.from(flatMask.map(BigInt)), [batchSize, maxLen]);
    }
    if (needsTokenTypeIds) {
      const flatTT = paddedTokenTypeIds.flat();
      feeds.token_type_ids = new ort.Tensor("int64", BigInt64Array.from(flatTT.map(BigInt)), [batchSize, maxLen]);
    }

    // Run inference
    const t0 = performance.now();
    const results = await session.run(feeds);
    const inferenceMs = performance.now() - t0;

    // Extract embeddings — try common output patterns
    let embeddings: number[][] = [];

    // Pattern 1: last_hidden_state [batch, seq, hidden]
    if (results.last_hidden_state) {
      const data = results.last_hidden_state.data as Float32Array;
      const dims = results.last_hidden_state.dims; // [batch, seq, hidden]
      const seqLen = Number(dims[1]);
      const hiddenDim = Number(dims[2]);
      embeddingDim = hiddenDim;

      for (let b = 0; b < batchSize; b++) {
        // Mean pool across sequence (excluding padding)
        const pooled = new Array(hiddenDim).fill(0);
        let validTokens = 0;
        for (let s = 0; s < seqLen; s++) {
          const idx = b * seqLen * hiddenDim + s * hiddenDim;
          if (paddedAttentionMask[b][s] > 0) {
            for (let h = 0; h < hiddenDim; h++) {
              pooled[h] += data[idx + h];
            }
            validTokens++;
          }
        }
        if (validTokens > 0) {
          for (let h = 0; h < hiddenDim; h++) pooled[h] /= validTokens;
        }
        embeddings.push(pooled);
      }
    }
    // Pattern 2: pooler_output [batch, hidden]
    else if (results.pooler_output) {
      const data = results.pooler_output.data as Float32Array;
      const dims = results.pooler_output.dims;
      const hiddenDim = Number(dims[dims.length - 1]);
      embeddingDim = hiddenDim;

      for (let b = 0; b < batchSize; b++) {
        const emb: number[] = [];
        for (let h = 0; h < hiddenDim; h++) {
          emb.push(data[b * hiddenDim + h]);
        }
        embeddings.push(emb);
      }
    }
    // Pattern 3: First output tensor
    else {
      const firstKey = Object.keys(results)[0];
      const tensor = results[firstKey];
      const data = tensor.data as Float32Array;
      const dims = tensor.dims;

      if (dims.length === 2) {
        // [batch, hidden]
        const hiddenDim = Number(dims[1]);
        embeddingDim = hiddenDim;
        for (let b = 0; b < batchSize; b++) {
          const emb: number[] = [];
          for (let h = 0; h < hiddenDim; h++) emb.push(data[b * hiddenDim + h]);
          embeddings.push(emb);
        }
      } else if (dims.length === 3) {
        // [batch, seq, hidden] — take [CLS] token (first)
        const seqLen = Number(dims[1]);
        const hiddenDim = Number(dims[2]);
        embeddingDim = hiddenDim;
        for (let b = 0; b < batchSize; b++) {
          const emb: number[] = [];
          for (let h = 0; h < hiddenDim; h++) emb.push(data[b * seqLen * hiddenDim + h]);
          embeddings.push(emb);
        }
      }
    }

    postMessage({
      type: "EMBEDDINGS",
      payload: {
        embeddings,
        embeddingDim,
        latencyMs: inferenceMs,
      },
    });
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    postMessage({ type: "ERROR", payload: { message: `Extraction failed: ${msg}` } });
  }
}
