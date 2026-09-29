/**
 * Запуск bridge-скрипта и кэш результата классификации.
 *
 * Классификация — целиком на стороне пайплайна columba (эвристика по ширине
 * кадра + CNN этапа 1 + арбитраж): интерфейс не дублирует логику, а только
 * показывает её результат.
 */

import { execFile } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
export const REPO_ROOT = path.resolve(HERE, "..", "..");
const BRIDGE = path.join(REPO_ROOT, "interface", "bridge", "describe.py");
const VENV_PYTHON = process.platform === "win32" ? ["Scripts", "python.exe"] : ["bin", "python"];
const PYTHON = process.env.COLUMBA_PYTHON ?? path.join(REPO_ROOT, ".venv", ...VENV_PYTHON);

/** Каталоги сканирования: подкаталог исследования = папка пациента, поэтому
 * трейн сканируем от «Исследования», а не от корня data/. */
const DEFAULT_ROOTS = [
  path.join(REPO_ROOT, "data", "НД_для_обучения", "Исследования"),
  path.join(REPO_ROOT, "data", "Для теста"),
];

/** Каталог загрузок из веб-интерфейса: каждая загрузка — своя папка-«исследование». */
export const UPLOAD_ROOT = path.join(REPO_ROOT, "data", "Загруженные");

export interface ManifestFile {
  file_id: string;
  file_name: string;
  relative_path: string;
  abs_path: string;
  study_folder: string;
  read_status: string;
  read_error: string | null;
  rows: number | null;
  cols: number | null;
  region: string;
  region_final: string;
  hip_side: string | null;
  hip_side_final: string | null;
  hip_side_score: number | null;
  hip_side_confident: boolean;
  hip_side_method: string | null;
  region_method: string;
  region_disagreement: boolean;
  cnn_label: string | null;
  dedup_group_id: string | null;
  dedup_group_size: number | null;
  /** Добавляется сервером: уникальный id и корень, из которого файл пришёл. */
  id: string;
  root: string;
}

export interface Manifest {
  generated_at: string;
  cnn_available: boolean;
  region_labels: Record<string, string>;
  pixel_spacing_mm: { y: number; x: number };
  roots: string[];
  files: ManifestFile[];
  warnings: string[];
}

export type ScanState =
  | { state: "idle" }
  | { state: "scanning"; started_at: string }
  | { state: "ready"; manifest: Manifest }
  | { state: "error"; message: string };

let current: ScanState = { state: "idle" };
let inFlight: Promise<Manifest> | null = null;

export function scanRoots(): string[] {
  const fromEnv = process.env.COLUMBA_SCAN_ROOTS;
  const roots = fromEnv ? fromEnv.split(path.delimiter) : DEFAULT_ROOTS.slice();
  if (!roots.includes(UPLOAD_ROOT)) roots.push(UPLOAD_ROOT);
  return roots.filter((root) => existsSync(root));
}

export function getState(): ScanState {
  return current;
}

export function ensureScan(force = false): Promise<Manifest> {
  if (!force && current.state === "ready") return Promise.resolve(current.manifest);
  if (inFlight) return inFlight;

  current = { state: "scanning", started_at: new Date().toISOString() };
  inFlight = runBridge(scanRoots())
    .then((manifest) => {
      current = { state: "ready", manifest };
      return manifest;
    })
    .catch((error: Error) => {
      current = { state: "error", message: error.message };
      throw error;
    })
    .finally(() => {
      inFlight = null;
    });
  return inFlight;
}

function runBridge(roots: string[]): Promise<Manifest> {
  if (roots.length === 0) {
    return Promise.reject(new Error("не найден ни один каталог со снимками (data/)"));
  }
  return new Promise((resolve, reject) => {
    execFile(
      PYTHON,
      [BRIDGE, ...roots],
      {
        maxBuffer: 256 * 1024 * 1024,
        cwd: REPO_ROOT,
        encoding: "utf8",
        env: { ...process.env, PYTHONIOENCODING: "utf-8", PYTHONUTF8: "1" },
      },
      (error, stdout, stderr) => {
        if (error) {
          reject(new Error(`bridge не отработал: ${error.message}\n${stderr.slice(0, 2000)}`));
          return;
        }
        try {
          const payload = JSON.parse(stdout) as {
            cnn_available: boolean;
            region_labels: Record<string, string>;
            pixel_spacing_mm: { y: number; x: number };
            roots: { root: string; files: Omit<ManifestFile, "id" | "root">[] }[];
          };
          const files: ManifestFile[] = [];
          payload.roots.forEach((chunk, rootIndex) => {
            for (const file of chunk.files) {
              files.push({ ...file, root: chunk.root, id: `r${rootIndex}-${file.file_id}` });
            }
          });
          // Предупреждения о расхождениях веток пайплайн пишет в stdout/stderr;
          // собираем их для показа в интерфейсе.
          const warnings = stderr
            .split("\n")
            .filter((line) => line.includes("ПРЕДУПРЕЖДЕНИЕ"))
            .slice(0, 50);
          resolve({
            generated_at: new Date().toISOString(),
            cnn_available: payload.cnn_available,
            region_labels: payload.region_labels,
            pixel_spacing_mm: payload.pixel_spacing_mm,
            roots,
            files,
            warnings,
          });
        } catch (parseError) {
          reject(new Error(`не удалось разобрать ответ bridge: ${String(parseError)}`));
        }
      },
    );
  });
}
