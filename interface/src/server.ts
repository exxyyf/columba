/**
 * Просмотрщик DXA-снимков: Express-сервер.
 *
 * Отдаёт статический фронтенд, JSON-манифест с классификацией региона
 * (из пайплайна columba через bridge) и пиксели снимков для канваса.
 * Только локальный запуск, наружу ничего не ходит.
 */

import path from "node:path";
import { fileURLToPath } from "node:url";
import express from "express";

import { readDicomPixels } from "./dicomPixels.js";
import { ensureScan, getState, scanRoots } from "./manifest.js";

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
