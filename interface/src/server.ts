/**
 * Просмотрщик DXA-снимков: Express-сервер.
 *
 * Отдаёт статический фронтенд, JSON-манифест с классификацией региона
 * (из пайплайна columba через bridge) и пиксели снимков для канваса.
 * Только локальный запуск, наружу ничего не ходит.
 */

import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import AdmZip from "adm-zip";
import express from "express";

import { readDicomPixels } from "./dicomPixels.js";
import { UPLOAD_ROOT, ensureScan, getState, scanRoots } from "./manifest.js";
import { API_URL, ensureQuality } from "./quality.js";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const PUBLIC_DIR = path.resolve(HERE, "..", "public");
const PORT = Number(process.env.PORT ?? 8425);

const app = express();
app.use(express.static(PUBLIC_DIR));

app.get("/api/status", (_req, res) => {
  const state = getState();
  if (state.state === "ready") {
    res.json({ state: "ready", files: state.manifest.files.length });
  } else {
    res.json(state);
  }
});

app.get("/api/manifest", async (req, res) => {
  try {
    const manifest = await ensureScan(req.query.force === "1");
    res.json(manifest);
  } catch (error) {
    res.status(500).json({ error: String(error instanceof Error ? error.message : error) });
  }
});

/**
 * Результаты контроля качества (пайплайн columba через API-сервис):
 * ленивый запуск при первом запросе после каждого скана, дальше — прогресс
 * и накопленные результаты. Клиент опрашивает, пока state === "running".
 */
app.get("/api/quality", async (_req, res) => {
  try {
    const manifest = await ensureScan();
    res.json(ensureQuality(manifest));
  } catch (error) {
    res.status(500).json({ error: String(error instanceof Error ? error.message : error) });
  }
});

/**
 * Относительный путь загрузки -> безопасный путь внутри папки партии.
 * Отсекает попытки выйти из каталога («..», абсолютные пути) и чистит
 * каждый сегмент от символов, недопустимых в именах файлов.
 */
function safeRelPath(raw: string): string | null {
  const parts = raw
    .replaceAll("\\", "/")
    .split("/")
    .filter((part) => part.length > 0);
  const clean: string[] = [];
  for (const part of parts) {
    if (part === "." || part === "..") return null;
    const name = part.replace(/[\\/:*?"<>|]/g, "_").trim();
    if (!name) return null;
    clean.push(name);
  }
  return clean.length > 0 ? clean.join("/") : null;
}

/** Метка партии из заголовка: одна загрузка (пачка/папка/архив) — одна папка. */
function batchLabel(req: express.Request): string {
  return (
    String(req.header("X-Upload-Batch") ?? "")
      .replace(/[^0-9A-Za-z_-]/g, "")
      .slice(0, 40) || "без_метки"
  );
}

/**
 * Загрузка собственного DICOM: тело запроса — файл целиком, имя в X-File-Name
 * (URL-кодированное), метка партии в X-Upload-Batch. Папочная загрузка
 * дополнительно передаёт X-Relative-Path — путь относительно выбранной папки
 * (без её имени, как в /predict/zip API): структура исследований сохраняется
 * на диске, и парное определение стороны бедра продолжает работать.
 * Классификацию запускает клиент пересканированием после последнего файла.
 */
app.post("/api/upload", express.raw({ type: () => true, limit: "128mb" }), (req, res) => {
  try {
    const rawRel = decodeURIComponent(
      String(req.header("X-Relative-Path") ?? req.header("X-File-Name") ?? ""),
    );
    const rel = safeRelPath(rawRel);
    if (!rel) {
      res.status(400).json({ error: "не передан корректный путь файла (X-Relative-Path / X-File-Name)" });
      return;
    }
    const body = req.body as Buffer;
    if (!Buffer.isBuffer(body) || body.length === 0) {
      res.status(400).json({ error: "пустое тело запроса" });
      return;
    }
    const target = path.join(UPLOAD_ROOT, `Загрузка_${batchLabel(req)}`, ...rel.split("/"));
    mkdirSync(path.dirname(target), { recursive: true });
    writeFileSync(target, body);
    res.json({ saved: path.relative(UPLOAD_ROOT, target) });
  } catch (error) {
    res.status(500).json({ error: String(error instanceof Error ? error.message : error) });
  }
});

/**
 * Загрузка ZIP-архива: тело запроса — архив целиком. Сервер распаковывает
 * только `.dcm`-файлы, сохраняя структуру подпапок; имя единственного
 * корневого каталога архива отбрасывается — то же соглашение, что у
 * `/predict/zip` API-сервиса (study_folder не включает имя корня).
 */
app.post("/api/upload/zip", express.raw({ type: () => true, limit: "512mb" }), (req, res) => {
  try {
    const body = req.body as Buffer;
    if (!Buffer.isBuffer(body) || body.length === 0) {
      res.status(400).json({ error: "пустое тело запроса" });
      return;
    }
    let zip: AdmZip;
    try {
      zip = new AdmZip(body);
    } catch {
      res.status(400).json({ error: "файл не распознан как ZIP-архив" });
      return;
    }
    const entries = zip
      .getEntries()
      .filter((entry) => !entry.isDirectory && /\.dcm$/i.test(entry.entryName));
    if (entries.length === 0) {
      res.status(400).json({ error: "в архиве нет .dcm-файлов" });
      return;
    }
    const rels = entries.map((entry) => entry.entryName.replaceAll("\\", "/"));
    const topLevels = new Set(rels.map((rel) => rel.split("/")[0]));
    const stripRoot = topLevels.size === 1 && rels.every((rel) => rel.includes("/"));
    const dir = path.join(UPLOAD_ROOT, `Загрузка_${batchLabel(req)}`);
    let saved = 0;
    const skipped: string[] = [];
    for (let i = 0; i < entries.length; i += 1) {
      const rel = safeRelPath(stripRoot ? rels[i].split("/").slice(1).join("/") : rels[i]);
      if (!rel) {
        skipped.push(entries[i].entryName);
        continue;
      }
      const target = path.join(dir, ...rel.split("/"));
      mkdirSync(path.dirname(target), { recursive: true });
      writeFileSync(target, entries[i].getData());
      saved += 1;
    }
    res.json({ saved, skipped });
  } catch (error) {
    res.status(500).json({ error: String(error instanceof Error ? error.message : error) });
  }
});

/**
 * Разбор одного снимка: прокси к `/visualize` API-сервиса columba.
 * `?format=png` — кадр с оверлеем ориентиров, `?format=json` — регион,
 * сторона и результаты чекеров структурировано. Файл уходит в сервис с
 * диска, клиент повторно ничего не загружает.
 */
app.get("/api/visualize/:id", async (req, res) => {
  try {
    const manifest = await ensureScan();
    const file = manifest.files.find((entry) => entry.id === req.params.id);
    if (!file) {
      res.status(404).json({ error: "неизвестный id снимка" });
      return;
    }
    const fmt = req.query.format === "json" ? "json" : "png";
    const form = new FormData();
    const body = new Uint8Array(readFileSync(file.abs_path));
    form.append("file", new Blob([body]), file.file_name);
    const upstream = await fetch(`${API_URL}/visualize?format=${fmt}`, {
      method: "POST",
      body: form,
    });
    if (!upstream.ok) {
      const text = await upstream.text().catch(() => upstream.statusText);
      res
        .status(502)
        .json({ error: `сервис визуализации ответил ${upstream.status}: ${text.slice(0, 300)}` });
      return;
    }
    if (fmt === "json") {
      res.json(await upstream.json());
      return;
    }
    res.setHeader("Content-Type", "image/png");
    res.setHeader("Cache-Control", "public, max-age=3600");
    res.end(Buffer.from(await upstream.arrayBuffer()));
  } catch (error) {
    res.status(500).json({ error: String(error instanceof Error ? error.message : error) });
  }
});

/** Пиксели одного снимка: бинарный поток яркостей + размеры в заголовках. */
app.get("/api/pixels/:id", async (req, res) => {
  try {
    const manifest = await ensureScan();
    const file = manifest.files.find((entry) => entry.id === req.params.id);
    if (!file) {
      res.status(404).json({ error: "неизвестный id снимка" });
      return;
    }
    const frame = readDicomPixels(file.abs_path);
    res.setHeader("Content-Type", "application/octet-stream");
    res.setHeader("X-Rows", String(frame.rows));
    res.setHeader("X-Cols", String(frame.cols));
    res.setHeader("Cache-Control", "public, max-age=3600");
    res.end(Buffer.from(frame.gray.buffer, frame.gray.byteOffset, frame.gray.byteLength));
  } catch (error) {
    res.status(500).json({ error: String(error instanceof Error ? error.message : error) });
  }
});

app.listen(PORT, "127.0.0.1", () => {
  console.log(`Columba viewer: http://127.0.0.1:${PORT}`);
  console.log(`Каталоги сканирования:\n  ${scanRoots().join("\n  ")}`);
  // Прогрев: классификация запускается сразу, а не при первом запросе.
  ensureScan().catch((error) => console.error("скан не удался:", error.message));
});
