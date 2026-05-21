/**
 * ============================================================================
 * FrugalSloth — Core Type Definitions
 * ============================================================================
 * Shared TypeScript interfaces used across the entire application.
 * These types define the contracts between UI components, workers,
 * IndexedDB storage, and the ONNX export pipeline.
 * ============================================================================
 */

// ---------------------------------------------------------------------------
// Column / Schema Types
// ---------------------------------------------------------------------------

/** Detected data type for a CSV column */
export type ColumnType = "numeric" | "categorical" | "text" | "datetime" | "unknown";

/** Metadata for a single column in the dataset */
export interface ColumnSchema {
  /** Original column header from CSV */
  name: string;
  /** Detected data type */
  type: ColumnType;
  /** Number of non-null values */
  count: number;
  /** Number of null/empty values */
  nullCount: number;
  /** For numeric: mean value */
  mean?: number;
  /** For numeric: standard deviation */
  std?: number;
  /** For numeric: minimum value */
  min?: number;
  /** For numeric: maximum value */
  max?: number;
  /** For categorical: unique values */
  categories?: string[];
  /** Sample values for preview */
  samples: (string | number | null)[];
}

/** Complete dataset schema inferred from CSV */
export interface DatasetSchema {
  /** All columns in order */
  columns: ColumnSchema[];
  /** Index of the target/prediction column */
  targetIndex: number;
  /** Detected ML task type */
  taskType: "regression" | "binary_classification" | "multi_class_classification";
  /** Number of data rows */
  rowCount: number;
  /** Hash of the dataset for caching */
  datasetHash: string;
}

// ---------------------------------------------------------------------------
// Training Configuration
// ---------------------------------------------------------------------------

/** Hyperparameters for model training */
export interface TrainingConfig {
  /** Learning rate (default: 0.001) */
  learningRate: number;
  /** Number of training epochs */
  epochs: number;
  /** Batch size for training */
  batchSize: number;
  /** Hidden layer sizes, e.g. [64, 32] for two hidden layers */
  hiddenLayers: number[];
  /** Optimizer algorithm */
  optimizer: "adam" | "sgd" | "rmsprop";
  /** Validation split ratio (0-1) */
  validationSplit: number;
  /** Random seed for reproducibility */
  seed: number;
  /** Early stopping patience (0 = disabled) */
  earlyStoppingPatience: number;
  /** Whether to use WebGPU (falls back to WebGL → CPU) */
  useWebGpu: boolean;
}

/** Default training configuration — matching early version for proper curves */
export const DEFAULT_TRAINING_CONFIG: TrainingConfig = {
  learningRate: 0.0001,
  epochs: 30,
  batchSize: 32,
  hiddenLayers: [64],
  optimizer: "adam",
  validationSplit: 0.2,
  seed: 42,
  earlyStoppingPatience: 10,
  useWebGpu: false,
};

// ---------------------------------------------------------------------------
// Training State / Progress
// ---------------------------------------------------------------------------

/** Status of the training process */
export type TrainingStatus = "idle" | "running" | "paused" | "completed" | "error";

/** A single epoch's metrics */
export interface EpochMetrics {
  epoch: number;
  loss: number;
  valLoss: number;
  accuracy?: number;
  valAccuracy?: number;
  mae?: number;
  valMae?: number;
}

/** Model architecture summary */
export interface ModelSummary {
  inputDim: number;
  outputDim: number;
  hiddenLayers: number[];
  totalParams: number;
  trainableParams: number;
}

/** Normalization statistics stored with the model */
export interface NormalizationStats {
  /** Feature index → { mean, std } for z-score normalization */
  numeric: Record<number, { mean: number; std: number }>;
  /** Categorical column index → { value → encoded index } */
  categorical: Record<number, Record<string, number>>;
  /** Label encoder: class name → index */
  labelMap: Record<string, number>;
  /** Reverse: index → class name */
  reverseLabelMap: Record<number, string>;
}

// ---------------------------------------------------------------------------
// Stored Model (IndexedDB)
// ---------------------------------------------------------------------------

/** A saved model entry in IndexedDB */
export interface StoredModel {
  /** Unique identifier */
  id: string;
  /** User-friendly name */
  name: string;
  /** When the model was created */
  createdAt: number;
  /** When the model was last modified */
  updatedAt: number;
  /** Dataset schema used for training */
  schema: DatasetSchema;
  /** Training configuration used */
  config: TrainingConfig;
  /** Model architecture summary */
  modelSummary: ModelSummary;
  /** Normalization statistics */
  normStats: NormalizationStats;
  /** Training history (metrics per epoch) */
  history: EpochMetrics[];
  /** Best validation loss achieved */
  bestValLoss: number;
  /** Whether this model can be resumed (has saved weights) */
  hasWeights: boolean;
  /** Serialized model weights (JSON for MLP, smaller than binary) */
  weightsJson?: string;
  /** Reservoir samples for catastrophic forgetting prevention */
  reservoirSamples?: number[][];
}

// ---------------------------------------------------------------------------
// Worker Messages
// ---------------------------------------------------------------------------

/** Base message format for all worker communications */
export interface WorkerMessage<T = unknown> {
  type: string;
  payload: T;
  /** Message ID for request/response correlation */
  id?: string;
}

/** Messages sent TO the training worker */
export type TrainingCommand =
  | { type: "INIT"; payload: { schema: DatasetSchema; config: TrainingConfig; rows: number[][]; labels: number[]; normStats: NormalizationStats } }
  | { type: "START"; payload: { epochs?: number } }
  | { type: "PAUSE"; payload: {} }
  | { type: "RESUME"; payload: {} }
  | { type: "STOP"; payload: {} }
  | { type: "GET_WEIGHTS"; payload: {} }
  | { type: "LOAD_WEIGHTS"; payload: { weightsJson: string } };

/** Messages received FROM the training worker */
export type TrainingEvent =
  | { type: "READY"; payload: { modelSummary: ModelSummary } }
  | { type: "EPOCH_END"; payload: EpochMetrics }
  | { type: "TRAIN_END"; payload: { finalLoss: number; bestValLoss: number; weightsJson: string } }
  | { type: "PAUSED"; payload: { epoch: number; weightsJson: string } }
  | { type: "ERROR"; payload: { message: string } }
  | { type: "WEIGHTS"; payload: { weightsJson: string } };

/** Messages sent TO the inference worker */
export type InferenceCommand =
  | { type: "INIT"; payload: { schema: DatasetSchema; config: Pick<TrainingConfig, "hiddenLayers">; weightsJson: string; normStats: NormalizationStats } }
  | { type: "PREDICT"; payload: { inputs: number[] } }
  | { type: "PREDICT_BATCH"; payload: { inputs: number[][] } };

/** Messages received FROM the inference worker */
export type InferenceEvent =
  | { type: "READY"; payload: { inputDim: number; outputDim: number } }
  | { type: "PREDICTION"; payload: { predictions: number[][]; latencyMs: number } }
  | { type: "ERROR"; payload: { message: string } };

// ---------------------------------------------------------------------------
// Export / Microservice
// ---------------------------------------------------------------------------

/** Export bundle configuration */
export interface ExportConfig {
  /** Model name */
  name: string;
  /** Quantization level */
  quantization: "float32" | "int8" | "int4";
  /** Whether to embed model as base64 in JS */
  embedModel: boolean;
}

/** Contents of the exported zip */
export interface ExportBundle {
  /** The ES module engine file */
  engineJs: string;
  /** The ONNX model file (if not embedded) */
  modelOnnx?: Uint8Array;
  /** README with usage instructions */
  readme: string;
  /** Total estimated size in bytes */
  estimatedSizeBytes: number;
}
