/**
 * ============================================================================
 * FrugalSloth — JSON Parser
 * ============================================================================
 * Parses JSON datasets (arrays of objects or arrays of arrays).
 * Produces the same output format as the CSV parser for compatibility.
 * ============================================================================
 */

import type { DatasetSchema, NormalizationStats } from "@/types/frugalsloth";
import { parseCSVString } from "./csvParser";

/** Result of parsing a JSON file */
export interface JSONParseResult {
  schema: DatasetSchema;
  rawRows: string[][];
  numericRows: number[][];
  labels: number[];
  normStats: NormalizationStats;
}

/**
 * Parse a JSON file. Accepts:
 * - Array of objects: [{"feature1": 1, "feature2": 2, "label": 0}, ...]
 * - Array of arrays: [[1, 2, 0], [3, 4, 1], ...]
 * - Object with "data" key: {"data": [...]}
 * - Object with "rows" key: {"rows": [...]}
 */
export function parseJSON(text: string): JSONParseResult {
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    throw new Error("Invalid JSON: could not parse file");
  }

  // Handle wrapped objects
  if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
    const obj = parsed as Record<string, unknown>;
    if (Array.isArray(obj.data)) parsed = obj.data;
    else if (Array.isArray(obj.rows)) parsed = obj.rows;
    else if (Array.isArray(obj.samples)) parsed = obj.samples;
    else {
      // Single object — wrap as array
      parsed = [obj];
    }
  }

  if (!Array.isArray(parsed)) {
    throw new Error("JSON must contain an array of objects or arrays");
  }

  const arr = parsed as unknown[];
  if (arr.length === 0) {
    throw new Error("JSON array is empty");
  }

  // Detect format: array of objects vs array of arrays
  const first = arr[0];

  if (Array.isArray(first)) {
    // Array of arrays — convert to CSV-like format
    const headers = arr[0] as string[];
    const dataRows = arr.slice(1) as string[][];
    // Build CSV string
    const csvLines = [
      headers.join(","),
      ...dataRows.map((row) => row.map(String).join(",")),
    ];
    return parseCSVString(csvLines.join("\n")) as JSONParseResult;
  }

  if (first && typeof first === "object") {
    // Array of objects — convert to CSV-like format
    const keys = Object.keys(first as object);
    if (keys.length === 0) {
      throw new Error("JSON objects have no keys");
    }

    const headers = keys;
    const csvLines = [
      headers.join(","),
      ...arr.map((obj) => {
        const record = obj as Record<string, unknown>;
        return headers.map((k) => {
          const v = record[k];
          if (v === null || v === undefined) return "";
          if (typeof v === "string" && v.includes(",")) return `"${v}"`;
          return String(v);
        }).join(",");
      }),
    ];
    return parseCSVString(csvLines.join("\n")) as JSONParseResult;
  }

  throw new Error("JSON must be an array of objects or an array of arrays");
}

/**
 * Parse a JSON file from a File object.
 */
export async function parseJSONFile(file: File): Promise<JSONParseResult> {
  const text = await file.text();
  return parseJSON(text);
}
