/**
 * ============================================================================
 * FrugalSloth — Inference Web Worker
 * ============================================================================
 * Dedicated worker for real-time predictions. Keeps a separate model instance
 * so inference never blocks training.
 *
 * Handles:
 *   - Model loading from serialized weights
 *   - Single predictions (for live slider demo)
 *   - Batch predictions (for CSV bulk inference)
 *   - Latency self-timing
 *
 * Messages: see types/frugalsloth.ts — InferenceCommand / InferenceEvent
 * ============================================================================
 */

import * as tf from "@tensorflow/tfjs";
import type { DatasetSchema, TrainingConfig } from "@/types/frugalsloth";

// ---------------------------------------------------------------------------
// Worker State
// ---------------------------------------------------------------------------

let model: tf.Sequential | null = null;
let schema: DatasetSchema | null = null;
let inputDim = 0;
let outputDim = 0;
let isClassification = false;

// ---------------------------------------------------------------------------
// Message Handler
// ---------------------------------------------------------------------------

self.onmessage = async (event: MessageEvent) => {
  const { type, payload } = event.data;

  try {
    switch (type) {
      case "INIT":
        await handleInit(payload);
        break;
      case "PREDICT":
        handlePredict(payload.inputs);
        break;
      case "PREDICT_BATCH":
        handlePredictBatch(payload.inputs);
        break;
      default:
        postMessage({ type: "ERROR", payload: { message: `Unknown command: ${type}` } });
    }
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    postMessage({ type: "ERROR", payload: { message: msg } });
  }
};

// ---------------------------------------------------------------------------
// INIT — Load model from serialized weights
// ---------------------------------------------------------------------------

async function handleInit(payload: {
  schema: DatasetSchema;
  config: Pick<TrainingConfig, "hiddenLayers">;
  weightsJson: string;
  outputDim: number;
}) {
  await tf.ready();

  schema = payload.schema;
  inputDim = schema.columns.filter((_, i) => i !== schema!.targetIndex).length;
  outputDim = payload.outputDim;
  isClassification = schema.taskType !== "regression";

  // Dispose old model
  if (model) {
    model.dispose();
    model = null;
  }

  // Build model architecture matching training
  const newModel = tf.sequential();
  const hiddenLayers = payload.config.hiddenLayers;

  for (let i = 0; i < hiddenLayers.length; i++) {
    newModel.add(
      tf.layers.dense({
        inputShape: i === 0 ? [inputDim] : undefined,
        units: hiddenLayers[i],
        activation: "relu",
      })
    );
  }

  const outputActivation =
    schema.taskType === "regression"
      ? "linear"
      : outputDim === 2
      ? "sigmoid"
      : "softmax";

  newModel.add(
    tf.layers.dense({
      units: outputDim === 2 ? 1 : outputDim,
      activation: outputActivation,
    })
  );

  // Load weights
  deserializeWeights(newModel, payload.weightsJson);

  model = newModel;

  postMessage({
    type: "READY",
    payload: { inputDim, outputDim },
  });
}

// ---------------------------------------------------------------------------
// Predict single sample (for live sandbox)
// ---------------------------------------------------------------------------

function handlePredict(inputs: number[]) {
  if (!model) {
    postMessage({ type: "ERROR", payload: { message: "Model not initialized" } });
    return;
  }

  const t0 = performance.now();
  const inputTensor = tf.tensor2d([inputs]);
  const outputTensor = model.predict(inputTensor) as tf.Tensor;
  const predictions = Array.from(outputTensor.dataSync());
  inputTensor.dispose();
  outputTensor.dispose();
  const latencyMs = performance.now() - t0;

  // Post-process
  const processed = postProcess(predictions);

  postMessage({
    type: "PREDICTION",
    payload: {
      predictions: [processed],
      latencyMs,
    },
  });
}

// ---------------------------------------------------------------------------
// Predict batch (for CSV bulk inference)
// ---------------------------------------------------------------------------

function handlePredictBatch(inputs: number[][]) {
  if (!model) {
    postMessage({ type: "ERROR", payload: { message: "Model not initialized" } });
    return;
  }

  const t0 = performance.now();
  const inputTensor = tf.tensor2d(inputs);
  const outputTensor = model.predict(inputTensor) as tf.Tensor;
  const flatOutput = Array.from(outputTensor.dataSync());
  inputTensor.dispose();
  outputTensor.dispose();
  const latencyMs = performance.now() - t0;

  // Reshape outputs
  const numSamples = inputs.length;
  const predictions: number[][] = [];
  for (let i = 0; i < numSamples; i++) {
    const start = i * outputDim;
    const end = start + outputDim;
    const row = flatOutput.slice(start, end);
    predictions.push(postProcess(row));
  }

  postMessage({
    type: "PREDICTION",
    payload: {
      predictions,
      latencyMs,
    },
  });
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function postProcess(raw: number[]): number[] {
  if (isClassification && outputDim > 2) {
    // Softmax output — return probabilities
    const exp = raw.map((v) => Math.exp(v - Math.max(...raw)));
    const sum = exp.reduce((a, b) => a + b, 0);
    return exp.map((v) => v / sum);
  }
  return raw;
}

function deserializeWeights(model: tf.Sequential, json: string): void {
  const weightArrays = JSON.parse(json) as number[][];
  let idx = 0;
  for (const layer of model.layers) {
    const expected = layer.getWeights();
    const newWeights: tf.Tensor[] = [];
    for (const exp of expected) {
      const data = new Float32Array(weightArrays[idx]);
      newWeights.push(tf.tensor(data, exp.shape));
      idx++;
    }
    layer.setWeights(newWeights);
  }
}


