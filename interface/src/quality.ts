/**
 * Оценка качества снимков через API-сервис columba (/predict).
 *
 * Viewer не дублирует логику чекеров: после сканирования каталогов файлы
 * пачками отправляются в сервис инференса, а результаты (quality_class,
 * quality_prob, violation_type) раздаются клиенту по /api/quality.
 *
 * Пачки собираются по исследованиям одного корня и уходят с относительными
 * путями (`paths`) — так сервис видит ту же структуру папок и парная логика
 * стороны бедра внутри исследования работает как при обычной загрузке.
 */

import { readFileSync } from "node:fs";

import type { Manifest, ManifestFile } from "./manifest.js";

const API_URL = process.env.COLUMBA_API_URL ?? "http://127.0.0.1:8000";
/** Ограничители пачки: файлов и суммарных байт на один запрос к /predict. */
const CHUNK_FILES = 48;
const CHUNK_BYTES = 48 * 1024 * 1024;

export interface QualityRow {
  anatomical_region: string;
  quality_class: number;
  quality_prob: number;
  violation_type: string;
}

export type QualityState =
  | { state: "idle" }
  | { state: "running"; done: number; total: number; results: Record<string, QualityRow> }
  | { state: "ready"; results: Record<string, QualityRow> }
  | { state: "error"; message: string; results: Record<string, QualityRow> };

let current: QualityState = { state: "idle" };
/** generated_at манифеста, для которого запущен/готов расчёт. */
let runningFor: string | null = null;

export function ensureQuality(manifest: Manifest): QualityState {
  if (runningFor !== manifest.generated_at) {
    runningFor = manifest.generated_at;
    const results: Record<string, QualityRow> = {};
    current = { state: "running", done: 0, total: manifest.files.length, results };
    void runQuality(manifest, results);
  }
  return current;
}

async function runQuality(manifest: Manifest, results: Record<string, QualityRow>): Promise<void> {
  const stamp = manifest.generated_at;
  try {
    for (const chunk of buildChunks(manifest.files)) {
      const rows = await predictChunk(chunk);
      if (runningFor !== stamp) return; // за время запроса случился rescan
      for (const [id, row] of rows) results[id] = row;
      const done = Math.min(
        manifest.files.length,
        (current.state === "running" ? current.done : 0) + chunk.length,
      );
      current = { state: "running", done, total: manifest.files.length, results };
    }
    current = { state: "ready", results };
  } catch (error) {
    if (runningFor !== stamp) return;
    current = {
      state: "error",
      message: String(error instanceof Error ? error.message : error),
      results,
    };
  }
}

/** Пачки не смешивают корни и не разрывают исследование между запросами. */
function buildChunks(files: ManifestFile[]): ManifestFile[][] {
  const byStudy = new Map<string, ManifestFile[]>();
  for (const file of files) {
    const key = `${file.root}\u0000${file.study_folder}`;
    const bucket = byStudy.get(key);
    if (bucket) bucket.push(file);
    else byStudy.set(key, [file]);
  }
  const chunks: ManifestFile[][] = [];
  let chunkRoot: string | null = null;
  let chunkFiles: ManifestFile[] = [];
  let chunkBytes = 0;
  const flush = () => {
    if (chunkFiles.length > 0) chunks.push(chunkFiles);
    chunkFiles = [];
    chunkBytes = 0;
  };
  for (const study of byStudy.values()) {
    const bytes = study.reduce((sum, f) => sum + guessSize(f), 0);
    const root = study[0].root;
    const overflow =
      chunkFiles.length + study.length > CHUNK_FILES || chunkBytes + bytes > CHUNK_BYTES;
    if (chunkFiles.length > 0 && (root !== chunkRoot || overflow)) flush();
    chunkRoot = root;
    chunkFiles.push(...study);
    chunkBytes += bytes;
  }
  flush();
  return chunks;
}

function guessSize(file: ManifestFile): number {
  // Точный размер не нужен: кадры одного аппарата почти одинаковы (~700 КБ).
  return (file.rows ?? 800) * (file.cols ?? 300) * 2 + 8 * 1024;
}

/** Один запрос к /predict; возвращает пары [id файла, строка результата]. */
async function predictChunk(files: ManifestFile[]): Promise<[string, QualityRow][]> {
  const form = new FormData();
  const byPath = new Map<string, string>();
  for (const file of files) {
    const relPosix = file.relative_path.replaceAll("\\", "/");
    byPath.set(relPosix, file.id);
    const body = new Uint8Array(readFileSync(file.abs_path));
    form.append("files", new Blob([body]), file.file_name);
    form.append("paths", relPosix);
  }
  const response = await fetch(`${API_URL}/predict?format=json`, { method: "POST", body: form });
  if (!response.ok) {
    const text = await response.text().catch(() => response.statusText);
    throw new Error(`сервис качества ответил ${response.status}: ${text.slice(0, 300)}`);
  }
  const payload = (await response.json()) as {
    rows: {
      relative_path: string;
      anatomical_region: string;
      quality_class: number;
      quality_prob: number;
      violation_type: string | null;
    }[];
  };
  const pairs: [string, QualityRow][] = [];
  for (const row of payload.rows) {
    const id = byPath.get(String(row.relative_path).replaceAll("\\", "/"));
    if (!id) continue;
    pairs.push([
      id,
      {
        anatomical_region: row.anatomical_region,
        quality_class: Number(row.quality_class),
        quality_prob: Number(row.quality_prob),
        violation_type: row.violation_type ?? "",
      },
    ]);
  }
  return pairs;
}
