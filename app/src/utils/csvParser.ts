/**
 * ============================================================================
 * FrugalSloth — CSV Parser with Intelligent Type Inference
 * ============================================================================
 * Wraps PapaParse for streaming CSV parsing and adds column type detection:
 *   - numeric: all values parse as valid numbers
 *   - categorical: ≤20 unique string values
 *   - text: long strings (>50 chars) or high cardinality
 *   - datetime: matches common date patterns
 *
 * Produces a DatasetSchema + preprocessed numeric rows ready for training.
 * ============================================================================
 */

import Papa from "papaparse";
import type { ColumnSchema, DatasetSchema, ColumnType, NormalizationStats } from "@/types/frugalsloth";

// ---------------------------------------------------------------------------
// Type Detection Heuristics
// ---------------------------------------------------------------------------

const SNIFF_ROW_COUNT = 200;
const CATEGORICAL_MAX_UNIQUE = 20;
const TEXT_MIN_LENGTH = 50;
const DATE_PATTERNS = [
  /^\d{4}-\d{2}-\d{2}/,           // 2024-01-15
  /^\d{2}\/\d{2}\/\d{4}/,          // 01/15/2024
  /^\d{2}-\d{2}-\d{4}/,           // 15-01-2024
  /^\d{4}\/\d{2}\/\d{2}/,          // 2024/01/15
];

/**
 * Try to parse a value as a number. Returns null if not a valid number.
 */
function tryParseNumber(value: string): number | null {
  const trimmed = value.trim();
  if (trimmed === "" || trimmed === "null" || trimmed === "NA" || trimmed === "N/A" || trimmed === "?") {
    return null;
  }
  const parsed = Number(trimmed);
  if (Number.isNaN(parsed) || !Number.isFinite(parsed)) {
    return null;
  }
  return parsed;
}

/**
 * Check if a string looks like a date.
 */
function looksLikeDate(value: string): boolean {
  const trimmed = value.trim();
  if (trimmed.length < 6) return false;
  return DATE_PATTERNS.some((pattern) => pattern.test(trimmed));
}

/**
 * Infer the type of a column from its sampled values.
 */
function inferColumnType(values: string[]): ColumnType {
  let numericCount = 0;
  let nullCount = 0;
  let dateCount = 0;
  let totalLength = 0;
  const uniqueValues = new Set<string>();

  for (const v of values) {
    const trimmed = v.trim();
    if (trimmed === "" || trimmed === "null" || trimmed === "NA" || trimmed === "N/A" || trimmed === "?") {
      nullCount++;
      continue;
    }
    totalLength += trimmed.length;
    uniqueValues.add(trimmed);

    if (tryParseNumber(trimmed) !== null) {
      numericCount++;
    }
    if (looksLikeDate(trimmed)) {
      dateCount++;
    }
  }

  const nonNullValues = values.length - nullCount;
  if (nonNullValues === 0) return "unknown";

  // If >80% look like dates → datetime
  if (dateCount / nonNullValues > 0.8) {
    return "datetime";
  }

  // If >80% parse as numbers → numeric
  if (numericCount / nonNullValues > 0.8) {
    return "numeric";
  }

  const avgLength = totalLength / nonNullValues;
  const uniqueCount = uniqueValues.size;

  // Long strings with many unique values → text
  if (avgLength > TEXT_MIN_LENGTH && uniqueCount > CATEGORICAL_MAX_UNIQUE) {
    return "text";
  }

  // Few unique values → categorical
  if (uniqueCount <= CATEGORICAL_MAX_UNIQUE && uniqueCount < nonNullValues * 0.5) {
    return "categorical";
  }

  // Default: text for high-cardinality string columns
  return "text";
}

/**
 * Compute column statistics from raw string values.
 */
function computeColumnStats(name: string, values: string[], type: ColumnType): ColumnSchema {
  const samples: (string | number | null)[] = [];
  let nullCount = 0;
  const numericValues: number[] = [];
  const uniqueSet = new Set<string>();

  for (const v of values) {
    const trimmed = v.trim();
    if (trimmed === "" || trimmed === "null" || trimmed === "NA" || trimmed === "N/A" || trimmed === "?") {
      nullCount++;
      samples.push(null);
      continue;
    }
    uniqueSet.add(trimmed);

    const num = tryParseNumber(trimmed);
    if (num !== null) {
      numericValues.push(num);
      samples.push(num);
    } else {
      samples.push(trimmed.substring(0, 50));
    }

    if (samples.length > 5) break; // keep only first 5 samples for preview
  }

  const col: ColumnSchema = {
    name,
    type,
    count: values.length - nullCount,
    nullCount,
    samples: samples.slice(0, 5),
  };

  if (type === "numeric" && numericValues.length > 0) {
    const sum = numericValues.reduce((a, b) => a + b, 0);
    col.mean = sum / numericValues.length;
    col.min = Math.min(...numericValues);
    col.max = Math.max(...numericValues);
    const variance = numericValues.reduce((a, b) => a + (b - col.mean!) ** 2, 0) / numericValues.length;
    col.std = Math.sqrt(variance);
  }

  if (type === "categorical") {
    col.categories = Array.from(uniqueSet).slice(0, CATEGORICAL_MAX_UNIQUE);
  }

  return col;
}

// ---------------------------------------------------------------------------
// Main Parse Function
// ---------------------------------------------------------------------------

/** Result of parsing a CSV file */
export interface ParseResult {
  schema: DatasetSchema;
  /** Raw parsed rows (string values) */
  rawRows: string[][];
  /** Preprocessed numeric rows (ready for training) */
  numericRows: number[][];
  /** Labels (target column extracted) */
  labels: number[];
  /** Normalization statistics */
  normStats: NormalizationStats;
}

/**
 * Compute a simple hash of the dataset for caching.
 */
function hashDataset(rows: string[][]): string {
  const str = rows.slice(0, 50).map((r) => r.join("|")).join("\n");
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash + char) | 0;
  }
  return Math.abs(hash).toString(36);
}

/**
 * Parse a CSV file and infer the complete schema.
 * Uses PapaParse for the heavy lifting, then runs type detection.
 */
export async function parseCSV(
  file: File,
  onProgress?: (percent: number) => void
): Promise<ParseResult> {
  return new Promise((resolve, reject) => {
    Papa.parse<string[]>(file, {
      complete: (results) => {
        try {
          const parseResult = processParsedData(results.data);
          resolve(parseResult);
        } catch (err) {
          reject(err);
        }
      },
      error: (error: Error) => reject(error),
      step: onProgress
        ? (results) => {
            const percent = Math.round((results.meta.cursor / file.size) * 100);
            if (percent % 10 === 0) onProgress(percent);
          }
        : undefined,
      skipEmptyLines: true,
      dynamicTyping: false, // We handle type conversion ourselves
    });
  });
}

/**
 * Parse CSV from a raw string (for demo datasets).
 */
export function parseCSVString(csvText: string): ParseResult {
  const results = Papa.parse<string[]>(csvText, {
    skipEmptyLines: true,
    dynamicTyping: false,
  });
  return processParsedData(results.data);
}

/**
 * Process raw parsed CSV data into a structured result.
 */
function processParsedData(data: string[][]): ParseResult {
  if (data.length < 2) {
    throw new Error("CSV must have at least a header row and one data row");
  }

  const headers = data[0];
  const rows = data.slice(1).filter((r) => r.some((cell) => cell.trim() !== ""));

  if (rows.length === 0) {
    throw new Error("No data rows found after header");
  }

  // Sample up to SNIFF_ROW_COUNT rows for type detection
  const sampleSize = Math.min(rows.length, SNIFF_ROW_COUNT);
  const sampledRows = rows.slice(0, sampleSize);

  // Infer column types
  const columns: ColumnSchema[] = headers.map((name, colIdx) => {
    const values = sampledRows.map((r) => r[colIdx] ?? "");
    const type = inferColumnType(values);
    return computeColumnStats(name, values, type);
  });

  // Default target: last numeric or categorical column
  let targetIndex = columns.length - 1;
  // Prefer the last numeric column if available
  for (let i = columns.length - 1; i >= 0; i--) {
    if (columns[i].type === "numeric" || columns[i].type === "categorical") {
      targetIndex = i;
      break;
    }
  }

  // Detect task type
  const targetCol = columns[targetIndex];
  let taskType: DatasetSchema["taskType"] = "regression";
  if (targetCol.type === "categorical" && targetCol.categories) {
    taskType = targetCol.categories.length === 2 ? "binary_classification" : "multi_class_classification";
  } else if (targetCol.type === "numeric") {
    // Check if it's actually integer categories
    const uniqueVals = new Set<number>();
    for (const r of sampledRows) {
      const n = tryParseNumber(r[targetIndex]);
      if (n !== null) uniqueVals.add(Math.round(n));
      if (uniqueVals.size > 20) break;
    }
    if (uniqueVals.size <= 2) {
      taskType = "binary_classification";
    } else if (uniqueVals.size <= 20) {
      taskType = "multi_class_classification";
    }
  }

  const schema: DatasetSchema = {
    columns,
    targetIndex,
    taskType,
    rowCount: rows.length,
    datasetHash: hashDataset(rows),
  };

  // Build normalization stats and preprocess rows
  const normStats = buildNormalizationStats(columns, targetIndex, rows);
  const { numericRows, labels } = preprocessRows(rows, columns, targetIndex, normStats, taskType);

  return {
    schema,
    rawRows: rows,
    numericRows,
    labels,
    normStats,
  };
}

// ---------------------------------------------------------------------------
// Normalization & Preprocessing
// ---------------------------------------------------------------------------

/**
 * Build normalization statistics from the dataset.
 */
function buildNormalizationStats(
  columns: ColumnSchema[],
  targetIndex: number,
  _rows: string[][]
): NormalizationStats {
  const numeric: NormalizationStats["numeric"] = {};
  const categorical: NormalizationStats["categorical"] = {};
  const labelMap: Record<string, number> = {};
  const reverseLabelMap: Record<number, string> = {};

  for (let colIdx = 0; colIdx < columns.length; colIdx++) {
    if (colIdx === targetIndex) continue;
    const col = columns[colIdx];

    if (col.type === "numeric" && col.mean !== undefined && col.std !== undefined && col.std > 0) {
      numeric[colIdx] = { mean: col.mean, std: col.std };
    }

    if (col.type === "categorical" && col.categories) {
      const mapping: Record<string, number> = {};
      col.categories.forEach((cat, i) => {
        mapping[cat] = i;
      });
      categorical[colIdx] = mapping;
    }
  }

  // Build label map for classification
  const targetCol = columns[targetIndex];
  if (targetCol.type === "categorical" && targetCol.categories) {
    targetCol.categories.forEach((cat, i) => {
      labelMap[cat] = i;
      reverseLabelMap[i] = cat;
    });
  }

  return { numeric, categorical, labelMap, reverseLabelMap };
}

/**
 * Preprocess raw CSV rows into numeric feature arrays and labels.
 */
function preprocessRows(
  rows: string[][],
  columns: ColumnSchema[],
  targetIndex: number,
  normStats: NormalizationStats,
  taskType: DatasetSchema["taskType"]
): { numericRows: number[][]; labels: number[] } {
  const numericRows: number[][] = [];
  const labels: number[] = [];

  for (const row of rows) {
    const features: number[] = [];
    let hasNull = false;

    for (let colIdx = 0; colIdx < columns.length; colIdx++) {
      const rawValue = row[colIdx]?.trim() ?? "";
      const isNull = rawValue === "" || rawValue === "null" || rawValue === "NA" || rawValue === "N/A" || rawValue === "?";

      if (colIdx === targetIndex) {
        // Extract label
        if (isNull) {
          hasNull = true;
          break;
        }
        const label = extractLabel(rawValue, columns[colIdx], taskType, normStats);
        if (label === null) {
          hasNull = true;
          break;
        }
        labels.push(label);
        continue;
      }

      // Extract feature
      if (isNull) {
        hasNull = true;
        break;
      }

      const feature = extractFeature(rawValue, columns[colIdx], colIdx, normStats);
      if (feature === null) {
        hasNull = true;
        break;
      }
      features.push(feature);
    }

    if (!hasNull && features.length > 0) {
      numericRows.push(features);
      // Ensure labels length matches
      if (labels.length !== numericRows.length) {
        numericRows.pop();
      }
    }
  }

  return { numericRows, labels };
}

/**
 * Extract a single feature value as a normalized number.
 */
function extractFeature(
  rawValue: string,
  col: ColumnSchema,
  colIdx: number,
  normStats: NormalizationStats
): number | null {
  if (col.type === "numeric") {
    const num = tryParseNumber(rawValue);
    if (num === null) return null;
    // Z-score normalize
    const stats = normStats.numeric[colIdx];
    if (stats && stats.std > 0) {
      return (num - stats.mean) / stats.std;
    }
    return num;
  }

  if (col.type === "categorical") {
    const mapping = normStats.categorical[colIdx];
    if (!mapping) return null;
    // One-hot would expand dimensionality; use index encoding for simplicity
    const idx = mapping[rawValue] ?? 0;
    return idx;
  }

  if (col.type === "datetime") {
    const ts = Date.parse(rawValue);
    if (Number.isNaN(ts)) return null;
    // Normalize to days since epoch, then z-score
    return ts / 86400000;
  }

  // Text columns: skip for MLP (handled by transformer path)
  return 0;
}

/**
 * Extract a label value as a numeric index.
 */
function extractLabel(
  rawValue: string,
  col: ColumnSchema,
  taskType: DatasetSchema["taskType"],
  normStats: NormalizationStats
): number | null {
  if (taskType === "regression") {
    return tryParseNumber(rawValue);
  }

  // Classification
  if (col.type === "categorical" && normStats.labelMap[rawValue] !== undefined) {
    return normStats.labelMap[rawValue];
  }

  // Numeric labels (e.g., 0/1)
  const num = tryParseNumber(rawValue);
  if (num !== null) {
    return Math.round(num);
  }

  return null;
}

/**
 * Get the input dimension (number of features) from processed rows.
 */
export function getInputDim(rows: number[][]): number {
  if (rows.length === 0) return 0;
  return rows[0].length;
}

/**
 * Get the output dimension (number of classes for classification, 1 for regression).
 */
export function getOutputDim(labels: number[], taskType: DatasetSchema["taskType"]): number {
  if (taskType === "regression") return 1;
  return Math.max(...labels) + 1;
}
