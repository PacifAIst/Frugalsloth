/**
 * ============================================================================
 * FrugalSloth — IndexedDB Persistence Layer
 * ============================================================================
 * Wraps the browser's IndexedDB API to store:
 *   - Trained model weights and architecture
 *   - Dataset schemas and normalization stats
 *   - Training history / run logs
 *   - Reservoir samples for catastrophic forgetting prevention
 *
 * Uses a single database "FrugalSlothDB" with object stores:
 *   - models: StoredModel entries (keyed by id)
 *   - datasets: Raw parsed dataset rows (keyed by hash)
 *   - config: App preferences and settings
 * ============================================================================
 */

import type { StoredModel, DatasetSchema } from "@/types/frugalsloth";

const DB_NAME = "FrugalSlothDB";
const DB_VERSION = 1;

/** Object store names */
const STORES = {
  models: "models",
  datasets: "datasets",
  config: "config",
} as const;

let dbPromise: Promise<IDBDatabase> | null = null;

/**
 * Get (or create) the IndexedDB database connection.
 * Lazy-initialized on first access.
 */
function getDB(): Promise<IDBDatabase> {
  if (dbPromise) return dbPromise;
  dbPromise = new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);
    request.onupgradeneeded = (event) => {
      const db = (event.target as IDBOpenDBRequest).result;
      // Models store: key = model id
      if (!db.objectStoreNames.contains(STORES.models)) {
        const modelStore = db.createObjectStore(STORES.models, { keyPath: "id" });
        modelStore.createIndex("createdAt", "createdAt", { unique: false });
        modelStore.createIndex("name", "name", { unique: false });
      }
      // Datasets store: key = dataset hash
      if (!db.objectStoreNames.contains(STORES.datasets)) {
        db.createObjectStore(STORES.datasets, { keyPath: "hash" });
      }
      // Config store: key = config key
      if (!db.objectStoreNames.contains(STORES.config)) {
        db.createObjectStore(STORES.config, { keyPath: "key" });
      }
    };
  });
  return dbPromise;
}

// ---------------------------------------------------------------------------
// Model CRUD
// ---------------------------------------------------------------------------

/**
 * Save a trained model (or update existing).
 */
export async function saveModel(model: StoredModel): Promise<void> {
  const db = await getDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORES.models, "readwrite");
    const store = tx.objectStore(STORES.models);
    const request = store.put(model);
    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error);
  });
}

/**
 * Load a model by its ID.
 */
export async function loadModel(id: string): Promise<StoredModel | null> {
  const db = await getDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORES.models, "readonly");
    const store = tx.objectStore(STORES.models);
    const request = store.get(id);
    request.onsuccess = () => resolve(request.result ?? null);
    request.onerror = () => reject(request.error);
  });
}

/**
 * List all saved models, ordered by creation date (newest first).
 */
export async function listModels(): Promise<StoredModel[]> {
  const db = await getDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORES.models, "readonly");
    const store = tx.objectStore(STORES.models);
    const index = store.index("createdAt");
    const request = index.openCursor(null, "prev");
    const models: StoredModel[] = [];
    request.onsuccess = () => {
      const cursor = request.result;
      if (cursor) {
        models.push(cursor.value);
        cursor.continue();
      } else {
        resolve(models);
      }
    };
    request.onerror = () => reject(request.error);
  });
}

/**
 * Delete a model by ID.
 */
export async function deleteModel(id: string): Promise<void> {
  const db = await getDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORES.models, "readwrite");
    const store = tx.objectStore(STORES.models);
    const request = store.delete(id);
    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error);
  });
}

/**
 * Count saved models.
 */
export async function countModels(): Promise<number> {
  const db = await getDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORES.models, "readonly");
    const store = tx.objectStore(STORES.models);
    const request = store.count();
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

// ---------------------------------------------------------------------------
// Dataset Cache
// ---------------------------------------------------------------------------

/** Cached dataset entry */
interface CachedDataset {
  hash: string;
  schema: DatasetSchema;
  rows: number[][];
  labels: number[];
  cachedAt: number;
}

/**
 * Save parsed dataset to cache.
 */
export async function cacheDataset(data: CachedDataset): Promise<void> {
  const db = await getDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORES.datasets, "readwrite");
    const store = tx.objectStore(STORES.datasets);
    const request = store.put(data);
    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error);
  });
}

/**
 * Load cached dataset by hash.
 */
export async function loadCachedDataset(hash: string): Promise<CachedDataset | null> {
  const db = await getDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORES.datasets, "readonly");
    const store = tx.objectStore(STORES.datasets);
    const request = store.get(hash);
    request.onsuccess = () => resolve(request.result ?? null);
    request.onerror = () => reject(request.error);
  });
}

// ---------------------------------------------------------------------------
// Config / Preferences
// ---------------------------------------------------------------------------

/**
 * Save a config key-value pair.
 */
export async function setConfig(key: string, value: unknown): Promise<void> {
  const db = await getDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORES.config, "readwrite");
    const store = tx.objectStore(STORES.config);
    const request = store.put({ key, value });
    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error);
  });
}

/**
 * Load a config value by key.
 */
export async function getConfig<T>(key: string, defaultValue: T): Promise<T> {
  const db = await getDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORES.config, "readonly");
    const store = tx.objectStore(STORES.config);
    const request = store.get(key);
    request.onsuccess = () => {
      resolve((request.result?.value as T) ?? defaultValue);
    };
    request.onerror = () => reject(request.error);
  });
}

// ---------------------------------------------------------------------------
// Storage Stats
// ---------------------------------------------------------------------------

/**
 * Get storage usage statistics for the FrugalSloth database.
 */
export async function getStorageStats(): Promise<{
  modelCount: number;
  datasetCount: number;
  estimatedBytes: number;
}> {
  const [modelCount] = await Promise.all([countModels()]);
  const db = await getDB();
  const datasetCount = await new Promise<number>((resolve, reject) => {
    const tx = db.transaction(STORES.datasets, "readonly");
    const store = tx.objectStore(STORES.datasets);
    const request = store.count();
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
  // Rough estimate: can't get exact bytes easily in IDB
  return {
    modelCount,
    datasetCount,
    estimatedBytes: 0, // Would need StorageManager API
  };
}

/**
 * Clear all FrugalSloth data (nuclear option).
 */
export async function clearAllData(): Promise<void> {
  dbPromise = null;
  return new Promise((resolve, reject) => {
    const request = indexedDB.deleteDatabase(DB_NAME);
    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error);
  });
}
