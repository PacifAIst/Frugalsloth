/**
 * FrugalSloth — Training Web Worker
 */

// Global error handler — fires BEFORE any imports, catches module load failures
(self as unknown as { onerror: ((e: ErrorEvent) => void) | null }).onerror = function(e) {
  const msg = e.message || (e.error && e.error.message) || String(e.error) || 'unknown';
  const line = e.lineno || '?';
  const col = e.colno || '?';
  (self as any).postMessage({ type: "ERROR", payload: { message: `Worker crash at ${line}:${col}: ${msg}` } });
};

import * as tf from "@tensorflow/tfjs";
import type { DatasetSchema, TrainingConfig, NormalizationStats, EpochMetrics, ModelSummary } from "@/types/frugalsloth";

// ---------------------------------------------------------------------------
// Worker State — module-level so START/RESUME can access
// ---------------------------------------------------------------------------

let currentModel: tf.Sequential | null = null;
let trainXs: tf.Tensor | null = null;
let trainYs: tf.Tensor | null = null;
let valXs: tf.Tensor | null = null;
let valYs: tf.Tensor | null = null;
let isPaused = false;
let shouldStop = false;
let currentEpoch = 0;
let totalEpochs = 0;
let currentSchema: DatasetSchema | null = null;
let currentConfig: TrainingConfig | null = null;
let bestWeights: string | null = null;
let bestValLoss = Infinity;
let patienceCounter = 0;
let modelSummary: ModelSummary | null = null;
let stoppedEpoch = 0; // actual epoch where training stopped (for early stopping)

// Reservoir sampling — prevents catastrophic forgetting during fine-tuning
// Stores ~5-10% of original training samples mixed into each batch
let reservoirXs: tf.Tensor | null = null;
let reservoirYs: tf.Tensor | null = null;
const RESERVOIR_RATIO = 0.08; // 8% of training data

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
      case "START":
        await handleStart(payload);
        break;
      case "PAUSE":
        handlePause();
        break;
      case "RESUME":
        await handleResume();
        break;
      case "STOP":
        handleStop();
        break;
      case "GET_WEIGHTS":
        handleGetWeights();
        break;
      case "LOAD_WEIGHTS":
        handleLoadWeights(payload);
        break;
      case "LOAD_RESERVOIR":
        handleLoadReservoir(payload);
        break;
      case "GET_RESERVOIR":
        handleGetReservoir();
        break;
      default:
        postError(`Unknown command: ${type}`);
    }
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    postError(msg);
  }
};

// ---------------------------------------------------------------------------
// INIT — Create model + tensors from data
// ---------------------------------------------------------------------------

async function handleInit(payload: {
  schema: DatasetSchema;
  config: TrainingConfig;
  rows: number[][];
  labels: number[];
  normStats: NormalizationStats;
}) {
  const { schema, config, rows, labels } = payload;

  try {
    // Cleanup previous state
    cleanup();
    trainStartTime = performance.now();

    currentSchema = schema;
    currentConfig = config;
    isPaused = false;
    shouldStop = false;
    currentEpoch = 0;
    stoppedEpoch = 0;
    totalEpochs = config.epochs;
    bestValLoss = Infinity;
    patienceCounter = 0;
    bestWeights = null;

    postDebug(`[Worker] INIT called: ${rows.length} rows, ${labels.length} labels, task=${schema.taskType}`);

    // TF.js backend setup
    postDebug(`[Worker] Calling tf.ready()...`);
    await tf.ready();
    postDebug(`[Worker] TF.js backend: ${tf.getBackend()}`);

    // Create tensors
    postDebug(`[Worker] Creating tensors...`);
    const allXs = tf.tensor2d(rows);
    postDebug(`[Worker] X tensor created: ${allXs.shape}`);
    const allYs = encodeLabels(labels, schema);
    postDebug(`[Worker] Y tensor created: ${allYs.shape}`);

  // Train/val split
  const n = rows.length;
  const valCount = Math.floor(n * config.validationSplit);
  const trainCount = n - valCount;
  postDebug(`[Worker] Split: train=${trainCount}, val=${valCount} (split=${config.validationSplit})`);

  if (valCount > 0) {
    // Random shuffle for split using Fisher-Yates
    const indices = Array.from({ length: n }, (_, i) => i);
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [indices[i], indices[j]] = [indices[j], indices[i]];
    }
    const trainIndices = indices.slice(0, trainCount);
    const valIndices = indices.slice(trainCount);

    trainXs = tf.gather(allXs, trainIndices);
    trainYs = tf.gather(allYs, trainIndices);
    valXs = tf.gather(allXs, valIndices);
    valYs = tf.gather(allYs, valIndices);
  } else {
    trainXs = allXs;
    trainYs = allYs;
  }

    allXs.dispose();
    allYs.dispose();
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    postDebug(`[Worker] INIT FAILED: ${msg}`);
    postError(`INIT failed: ${msg}`);
  }

  // Build MLP
  const inputDim = rows[0]?.length ?? 0;
  // Remap labels to sequential 0-indexed IDs for correct outputDim
  const uniqueLabels = Array.from(new Set(labels)).sort((a, b) => a - b);
  const outputDim = schema.taskType === "regression" ? 1 : uniqueLabels.length;
  postDebug(`[Worker] Model: input=${inputDim}, output=${outputDim} (from ${uniqueLabels.length} unique classes), hidden=[${config.hiddenLayers.join(",")}]`);

  const model = tf.sequential();

  for (let i = 0; i < config.hiddenLayers.length; i++) {
    model.add(
      tf.layers.dense({
        inputShape: i === 0 ? [inputDim] : undefined,
        units: config.hiddenLayers[i],
        activation: "relu",
        kernelInitializer: "glorotUniform",
        kernelRegularizer: tf.regularizers.l2({ l2: 0.01 }),
      })
    );
  }

  const outputActivation =
    schema.taskType === "regression"
      ? "linear"
      : outputDim === 2
      ? "sigmoid"
      : "softmax";

  model.add(
    tf.layers.dense({
      units: outputDim === 2 ? 1 : outputDim,
      activation: outputActivation,
    })
  );

  // Compile
  const loss =
    schema.taskType === "regression"
      ? "meanSquaredError"
      : outputDim === 2
      ? "binaryCrossentropy"
      : "categoricalCrossentropy";

  const optimizer =
    config.optimizer === "sgd"
      ? tf.train.sgd(config.learningRate)
      : config.optimizer === "rmsprop"
      ? tf.train.rmsprop(config.learningRate)
      : tf.train.adam(config.learningRate);

  const metrics: string[] = schema.taskType === "regression" ? ["mse"] : ["accuracy"];
  model.compile({ optimizer, loss, metrics });

  currentModel = model;
  postDebug(`[Worker] Model compiled: ${model.countParams()} params, loss=${loss}, opt=${config.optimizer}`);

  // Model summary
  const totalParams = model.countParams();
  modelSummary = {
    inputDim,
    outputDim,
    hiddenLayers: [...config.hiddenLayers],
    totalParams,
    trainableParams: totalParams,
  };

  postMessage({ type: "READY", payload: { modelSummary } });
}

// ---------------------------------------------------------------------------
// START — Run training loop
// ---------------------------------------------------------------------------

async function handleStart(payload: { epochs?: number }) {
  if (!currentModel || !trainXs || !trainYs || !currentSchema || !currentConfig) {
    postError("Model not initialized. Call INIT first.");
    return;
  }

  totalEpochs = payload.epochs ?? currentConfig.epochs;
  shouldStop = false;
  isPaused = false;
  trainStartTime = performance.now();
  postDebug(`[Worker] START: totalEpochs=${totalEpochs}, currentEpoch=${currentEpoch}, earlyStopPatience=${currentConfig.earlyStoppingPatience}`);

  // Continue from current epoch
  const remainingEpochs = totalEpochs - currentEpoch;
  if (remainingEpochs <= 0) {
    const elapsed = ((performance.now() - trainStartTime) / 1000).toFixed(1);
    postMessage({
      type: "TRAIN_END",
      payload: {
        finalLoss: bestValLoss,
        bestValLoss,
        weightsJson: bestWeights ?? serializeWeights(currentModel),
        elapsed,
      },
    });
    return;
  }

  try {
    // Mix reservoir samples to prevent catastrophic forgetting
    const [mixedXs, mixedYs] = mixReservoir(trainXs, trainYs);

    await currentModel.fit(mixedXs, mixedYs, {
      epochs: remainingEpochs,
      batchSize: currentConfig.batchSize,
      validationData: valXs && valYs ? [valXs, valYs] : undefined,
      verbose: 0,
      callbacks: {
        onEpochEnd: (epoch: number, logs?: tf.Logs) => {
          if (!logs) return;
          if (shouldStop) {
            currentModel!.stopTraining = true;
            return;
          }

          const actualEpoch = currentEpoch + epoch + 1;
          const loss = logs.loss as number;
          const valLoss = (logs.val_loss as number) ?? loss;

          // Raw debug for first 3 epochs: show actual loss + predictions
          if (actualEpoch <= 3) {
            postDebug(`[Worker] Epoch ${actualEpoch} RAW: loss=${loss.toPrecision(6)}, val=${valLoss.toPrecision(6)}`);
          }

          // Build metrics
          const metrics: EpochMetrics = {
            epoch: actualEpoch,
            loss,
            valLoss,
          };

          if (currentSchema!.taskType !== "regression") {
            metrics.accuracy = (logs.acc as number) ?? (logs.accuracy as number);
            metrics.valAccuracy = (logs.val_acc as number) ?? (logs.val_accuracy as number);
          } else {
            metrics.mae = (logs.mae as number);
            metrics.valMae = (logs.val_mae as number);
          }

          postMessage({ type: "EPOCH_END", payload: { ...metrics, totalEpochs } });

          // Early stopping check
          if (valLoss < bestValLoss) {
            bestValLoss = valLoss;
            bestWeights = serializeWeights(currentModel!);
            patienceCounter = 0;
          } else {
            patienceCounter++;
            if (currentConfig!.earlyStoppingPatience > 0 &&
                patienceCounter >= currentConfig!.earlyStoppingPatience) {
              stoppedEpoch = actualEpoch;
              postDebug(`[Worker] EARLY STOP at epoch ${actualEpoch}: no improvement for ${currentConfig!.earlyStoppingPatience} epochs`);
              currentModel!.stopTraining = true;
            }
          }
        },
        onTrainEnd: () => {
          const finalEpoch = stoppedEpoch > 0 ? stoppedEpoch : (currentEpoch + remainingEpochs);
          currentEpoch = finalEpoch;
          const finalWeights = bestWeights ?? serializeWeights(currentModel!);
          const elapsed = ((performance.now() - trainStartTime) / 1000).toFixed(1);
          postDebug(`[Worker] TRAIN_END: ${finalEpoch}/${totalEpochs} epochs in ${elapsed}s`);
          postMessage({
            type: "TRAIN_END",
            payload: {
              finalLoss: bestValLoss,
              bestValLoss,
              weightsJson: finalWeights,
              elapsed,
              epoch: finalEpoch,
              totalEpochs,
            },
          });
        },
      },
    });

    // TRAIN_END is handled in onTrainEnd callback above
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    postError(`Training failed: ${msg}`);
  }
}

// ---------------------------------------------------------------------------
// PAUSE / RESUME / STOP
// ---------------------------------------------------------------------------

function handlePause() {
  shouldStop = true;
  isPaused = true;
  if (currentModel) {
    currentModel.stopTraining = true;
  }
}

async function handleResume() {
  if (!isPaused) return;
  isPaused = false;
  shouldStop = false;
  // Re-run START from current epoch
  await handleStart({ epochs: totalEpochs });
}

function handleStop() {
  shouldStop = true;
  isPaused = false;
  if (currentModel) {
    currentModel.stopTraining = true;
  }
  postMessage({
    type: "TRAIN_END",
    payload: {
      finalLoss: bestValLoss,
      bestValLoss,
      weightsJson: bestWeights ?? (currentModel ? serializeWeights(currentModel) : ""),
    },
  });
}

// ---------------------------------------------------------------------------
// Weight Serialization
// ---------------------------------------------------------------------------

function handleGetWeights() {
  if (!currentModel) {
    postError("No model available.");
    return;
  }
  const weightsJson = bestWeights ?? serializeWeights(currentModel);
  postMessage({ type: "WEIGHTS", payload: { weightsJson } });
}

function handleLoadWeights(payload: { weightsJson: string }) {
  if (!currentModel) {
    postError("No model to load weights into.");
    return;
  }
  deserializeWeights(currentModel, payload.weightsJson);
  bestWeights = payload.weightsJson;
  postMessage({ type: "READY", payload: { loaded: true } });
}

// ---------------------------------------------------------------------------
// Reservoir Sampling — Catastrophic Forgetting Prevention
// ---------------------------------------------------------------------------

function handleLoadReservoir(payload: {
  samples: number[][];
  labels: number[];
}) {
  // Dispose old reservoir
  if (reservoirXs) { reservoirXs.dispose(); reservoirXs = null; }
  if (reservoirYs) { reservoirYs.dispose(); reservoirYs = null; }

  if (payload.samples.length > 0) {
    reservoirXs = tf.tensor2d(payload.samples);
    const schema = currentSchema;
    if (schema && schema.taskType !== "regression") {
      const uniqueLabels = Array.from(new Set(payload.labels)).sort((a, b) => a - b);
      const labelMap = new Map<number, number>();
      uniqueLabels.forEach((v, i) => labelMap.set(v, i));
      const remapped = payload.labels.map((l) => labelMap.get(l)!);
      const numClasses = uniqueLabels.length;
      reservoirYs = tf.oneHot(tf.tensor1d(remapped, "int32"), numClasses);
    } else {
      reservoirYs = tf.tensor2d(payload.labels.map((l) => [l]));
    }
    postMessage({
      type: "READY",
      payload: { reservoirLoaded: true, count: payload.samples.length },
    });
  }
}

function handleGetReservoir() {
  // Extract current training data as reservoir samples
  if (!trainXs || !trainYs) {
    postMessage({ type: "RESERVOIR", payload: { samples: [], labels: [] } });
    return;
  }
  const xsData = Array.from(trainXs.dataSync());
  const ysData = Array.from(trainYs.dataSync());
  const inputDim = trainXs.shape[1] ?? 0;
  const numSamples = trainXs.shape[0] ?? 0;
  const reservoirSize = Math.max(1, Math.floor(numSamples * RESERVOIR_RATIO));

  // Random sample
  const indices: number[] = [];
  for (let i = 0; i < numSamples; i++) indices.push(i);
  for (let i = indices.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }
  const selected = indices.slice(0, reservoirSize);

  const samples: number[][] = [];
  const labels: number[] = [];
  for (const idx of selected) {
    const rowStart = idx * inputDim;
    samples.push(xsData.slice(rowStart, rowStart + inputDim));
    // For classification, ys is one-hot; take argmax
    if (currentSchema && currentSchema.taskType !== "regression") {
      const numClasses = trainYs.shape[1] ?? 1;
      const labelStart = idx * numClasses;
      const oneHot = ysData.slice(labelStart, labelStart + numClasses);
      labels.push(oneHot.indexOf(Math.max(...oneHot)));
    } else {
      labels.push(ysData[idx]);
    }
  }

  postMessage({ type: "RESERVOIR", payload: { samples, labels } });
}

function mixReservoir(trainX: tf.Tensor, trainY: tf.Tensor): [tf.Tensor, tf.Tensor] {
  if (!reservoirXs || !reservoirYs) return [trainX, trainY];

  // Concatenate training data with reservoir samples
  const mixedX = tf.concat([trainX, reservoirXs], 0);
  const mixedY = tf.concat([trainY, reservoirYs], 0);
  return [mixedX, mixedY];
}

function serializeWeights(model: tf.Sequential): string {
  const arrays: number[][] = [];
  for (const layer of model.layers) {
    for (const w of layer.getWeights()) {
      arrays.push(Array.from(w.dataSync()));
    }
  }
  return JSON.stringify(arrays);
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

// ---------------------------------------------------------------------------
// Label Encoding
// ---------------------------------------------------------------------------

function encodeLabels(labels: number[], schema: DatasetSchema): tf.Tensor {
  if (schema.taskType === "regression") {
    return tf.tensor2d(labels.map((l) => [l]));
  }
  // Remap labels to sequential 0-indexed IDs
  const uniqueLabels = Array.from(new Set(labels)).sort((a, b) => a - b);
  const labelMap = new Map<number, number>();
  uniqueLabels.forEach((v, i) => labelMap.set(v, i));
  const remapped = labels.map((l) => labelMap.get(l)!);
  const numClasses = uniqueLabels.length;
  postDebug(`[Worker] Labels remapped: ${numClasses} classes [${uniqueLabels.join(",")}] → [${Array.from(new Set(remapped)).join(",")}]`);
  return tf.oneHot(tf.tensor1d(remapped, "int32"), numClasses);
}

// ---------------------------------------------------------------------------
// Cleanup
// ---------------------------------------------------------------------------

function cleanup() {
  if (currentModel) { currentModel.dispose(); currentModel = null; }
  if (trainXs) { trainXs.dispose(); trainXs = null; }
  if (trainYs) { trainYs.dispose(); trainYs = null; }
  if (valXs) { valXs.dispose(); valXs = null; }
  if (valYs) { valYs.dispose(); valYs = null; }
  if (reservoirXs) { reservoirXs.dispose(); reservoirXs = null; }
  if (reservoirYs) { reservoirYs.dispose(); reservoirYs = null; }
}

let trainStartTime = 0;

function postError(message: string) {
  postMessage({ type: "ERROR", payload: { message } });
}

function postDebug(message: string) {
  postMessage({ type: "DEBUG", payload: { message } });
}
