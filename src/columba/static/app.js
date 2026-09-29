// columba — этап 8, бонус 2 + этап 9, п. 9.6: веб-интерфейс поверх сервиса
// этапа 7. Никакой логики предсказания здесь нет — только вызовы
// /predict, /predict/zip и /visualize (columba/service.py) и отрисовка
// ответа.

const fileInput = document.getElementById("file-input");
const folderInput = document.getElementById("folder-input");
const zipInput = document.getElementById("zip-input");
const analyzeBtn = document.getElementById("analyze-btn");
const statusEl = document.getElementById("status");
const resultsPanel = document.getElementById("results-panel");
const resultsBody = document.querySelector("#results-table tbody");
const summaryEl = document.getElementById("summary");
const downloadCsvBtn = document.getElementById("download-csv-btn");
const vizPanel = document.getElementById("viz-panel");
const vizImage = document.getElementById("viz-image");
const vizInfo = document.getElementById("viz-info");

// Режим загрузки — ровно один из трёх источников активен одновременно
// (выбор одного сбрасывает два других, см. `resetSelection`).
const MODE_FLAT = "flat";
const MODE_FOLDER = "folder";
const MODE_ZIP = "zip";

// Имя/относительный путь -> File (для повторной отправки на /visualize без
// повторной загрузки пользователем). Пусто в режиме zip — сервер не хранит
// файлы между запросами (решение этапа 8), а из ZIP-архива, который ушёл на
// сервер целиком, отдельные File-объекты на клиенте недоступны.
let selectedFiles = new Map();
let selectedZip = null;
let uploadMode = null;
let lastRows = [];

function resetSelection() {
  selectedFiles = new Map();
  selectedZip = null;
}

fileInput.addEventListener("change", () => {
  resetSelection();
  uploadMode = MODE_FLAT;
  const seen = new Map();
  for (const file of fileInput.files) {
    selectedFiles.set(dedupeName(file.name, seen), file);
  }
  folderInput.value = "";
  zipInput.value = "";
  afterSelectionChange();
});

folderInput.addEventListener("change", () => {
  resetSelection();
  uploadMode = MODE_FOLDER;
  const seen = new Map();
  for (const file of folderInput.files) {
    if (!/\.dcm$/i.test(file.name)) continue; // папка может содержать не только .dcm
    // `webkitRelativePath` включает имя САМОЙ выбранной папки первым
    // сегментом — отрезаем его, чтобы `study_folder` на сервере совпадал с
    // подпапками, которые реально видит пользователь (то же соглашение,
    // что и у /predict/zip: имя корня архива не входит в relative_path).
    const parts = file.webkitRelativePath.split("/");
    const relPath = parts.length > 1 ? parts.slice(1).join("/") : file.name;
    selectedFiles.set(dedupeName(relPath, seen), file);
  }
  fileInput.value = "";
  zipInput.value = "";
  afterSelectionChange();
});

zipInput.addEventListener("change", () => {
  resetSelection();
  selectedZip = zipInput.files[0] || null;
  uploadMode = selectedZip ? MODE_ZIP : null;
  fileInput.value = "";
  folderInput.value = "";
  afterSelectionChange();
});

function afterSelectionChange() {
  const hasSelection = uploadMode === MODE_ZIP ? !!selectedZip : selectedFiles.size > 0;
  analyzeBtn.disabled = !hasSelection;
  document.querySelectorAll("[data-upload-mode]").forEach((field) => {
    field.classList.toggle("selected", hasSelection && field.dataset.uploadMode === uploadMode);
  });
  if (!hasSelection) {
    setStatus("");
  } else if (uploadMode === MODE_ZIP) {
    setStatus(`Выбран архив: ${selectedZip.name}`);
  } else {
    setStatus(`Выбрано файлов: ${selectedFiles.size}`);
  }
}

// Тот же алгоритм разрешения коллизий, что `_dedupe_filename` в
// service.py, — по posix-пути целиком (сохраняет каталог в имени
// коллизии, не только последний компонент): клиент и сервер видят файлы в
// одном порядке (FastAPI сохраняет порядок частей формы с одинаковым
// именем поля), поэтому независимо вычисленные уникальные имена совпадают
// без обмена дополнительными данными, и таблица результатов
// (`row.file_name`) остаётся кликабельной в интерфейсе.
function dedupeName(name, seen) {
  const count = seen.get(name) || 0;
  seen.set(name, count + 1);
  if (count === 0) return name;
  const slashIndex = name.lastIndexOf("/");
  const dir = slashIndex >= 0 ? name.slice(0, slashIndex + 1) : "";
  const leaf = slashIndex >= 0 ? name.slice(slashIndex + 1) : name;
  const dotIndex = leaf.lastIndexOf(".");
  if (dotIndex <= 0) return `${dir}${leaf}__${count}`;
  return `${dir}${leaf.slice(0, dotIndex)}__${count}${leaf.slice(dotIndex)}`;
}

analyzeBtn.addEventListener("click", async () => {
  if (uploadMode === MODE_ZIP && !selectedZip) return;
  if (uploadMode !== MODE_ZIP && selectedFiles.size === 0) return;

  setStatus(`Обработка (${modeLabel(uploadMode)})…`);
  analyzeBtn.disabled = true;

  try {
    let response;
    if (uploadMode === MODE_ZIP) {
      const formData = new FormData();
      formData.append("file", selectedZip);
      response = await fetch("/predict/zip?format=json", { method: "POST", body: formData });
    } else {
      const formData = new FormData();
      for (const [key, file] of selectedFiles.entries()) {
        formData.append("files", file);
        if (uploadMode === MODE_FOLDER) formData.append("paths", key);
      }
      response = await fetch("/predict?format=json", { method: "POST", body: formData });
    }
    if (!response.ok) {
      const detail = await safeErrorDetail(response);
      throw new Error(detail);
    }
    const body = await response.json();
    lastRows = body.rows;
    renderResults(body.rows);
    renderSummary(body.rows);
    setStatus(`Готово: ${body.n_files} файл(ов) за ${body.elapsed_seconds.toFixed(2)} с.`);
  } catch (err) {
    setStatus(`Ошибка: ${err.message}`, true);
  } finally {
    analyzeBtn.disabled = false;
  }
});

function modeLabel(mode) {
  if (mode === MODE_FOLDER) return "папка";
  if (mode === MODE_ZIP) return "zip";
  return "файлы";
}

function setStatus(text, isError = false) {
  statusEl.textContent = text;
  statusEl.classList.toggle("error", isError);
}

async function safeErrorDetail(response) {
  try {
    const body = await response.json();
    return body.detail || `HTTP ${response.status}`;
  } catch {
    return `HTTP ${response.status}`;
  }
}

function renderResults(rows) {
  resultsBody.innerHTML = "";
  for (const row of rows) {
    const tr = document.createElement("tr");

    const isViolation = Number(row.quality_class) === 1;
    const badgeClass = isViolation ? "violation" : "ok";
    const badgeText = isViolation ? "Нарушение" : "Норма";

    tr.innerHTML = `
      <td>${escapeHtml(row.file_name)}</td>
      <td>${escapeHtml(row.anatomical_region || "")}</td>
      <td><span class="badge ${badgeClass}">${badgeText}</span></td>
      <td>${escapeHtml(row.violation_type || "—")}</td>
      <td>${Number(row.quality_prob).toFixed(3)}</td>
      <td></td>
    `;

    const vizCell = tr.lastElementChild;
    const file = selectedFiles.get(row.file_name);
    if (file) {
      const btn = document.createElement("button");
      btn.className = "secondary";
      btn.textContent = "Посмотреть";
      btn.addEventListener("click", () => showVisualization(file));
      vizCell.appendChild(btn);
    } else {
      // Режим zip — исходный File недоступен на клиенте (решение этапа 8:
      // сервер не хранит файлы между запросами), кнопка честно неактивна,
      // а не просто отсутствует.
      const btn = document.createElement("button");
      btn.className = "secondary";
      btn.disabled = true;
      btn.textContent = "Недоступно";
      btn.setAttribute("aria-label", "Визуализация недоступна для загрузки из zip-архива");
      btn.title = "Недоступно для zip-загрузки — сервер не хранит файлы между запросами";
      vizCell.appendChild(btn);
    }

    resultsBody.appendChild(tr);
  }
  resultsPanel.hidden = false;
  downloadCsvBtn.hidden = rows.length === 0;
}

function renderSummary(rows) {
  if (rows.length === 0) {
    summaryEl.hidden = true;
    return;
  }
  const total = rows.length;
  const violations = rows.filter((row) => Number(row.quality_class) === 1).length;
  const byRegion = new Map();
  for (const row of rows) {
    const region = row.anatomical_region || "—";
    const stats = byRegion.get(region) || { total: 0, violations: 0 };
    stats.total += 1;
    if (Number(row.quality_class) === 1) stats.violations += 1;
    byRegion.set(region, stats);
  }
  const regionText = Array.from(byRegion.entries())
    .map(([region, stats]) => `${region} — ${stats.violations}/${stats.total}`)
    .join("; ");
  const pct = total > 0 ? Math.round((violations / total) * 100) : 0;
  summaryEl.textContent = `Всего файлов: ${total}. Нарушений: ${violations} (${pct}%). ${regionText}`;
  summaryEl.hidden = false;
}

downloadCsvBtn.addEventListener("click", () => {
  if (lastRows.length === 0) return;
  const csv = rowsToCsv(lastRows);
  const blob = new Blob(["﻿" + csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "columba_results.csv";
  a.click();
  URL.revokeObjectURL(url);
});

function rowsToCsv(rows) {
  const columns = Object.keys(rows[0]);
  const lines = [columns.map(csvField).join(",")];
  for (const row of rows) {
    lines.push(columns.map((column) => csvField(row[column])).join(","));
  }
  return lines.join("\r\n");
}

function csvField(value) {
  const text = value === null || value === undefined ? "" : String(value);
  return /["\n,]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

async function showVisualization(file) {
  setStatus(`Строим визуализацию для ${file.name}…`);
  vizInfo.innerHTML = "";
  try {
    const [pngResponse, jsonResponse] = await Promise.all([
      postVisualize(file, "png"),
      postVisualize(file, "json"),
    ]);
    if (!pngResponse.ok) throw new Error(await safeErrorDetail(pngResponse));
    if (!jsonResponse.ok) throw new Error(await safeErrorDetail(jsonResponse));

    const blob = await pngResponse.blob();
    const url = URL.createObjectURL(blob);
    if (vizImage.src) URL.revokeObjectURL(vizImage.src);
    vizImage.src = url;

    const info = await jsonResponse.json();
    vizImage.alt = describeInfoForAlt(file.name, info);
    renderVizInfo(info);

    vizPanel.hidden = false;
    vizPanel.scrollIntoView({ behavior: "smooth", block: "nearest" });
    setStatus("");
  } catch (err) {
    setStatus(`Ошибка визуализации: ${err.message}`, true);
  }
}

function postVisualize(file, format) {
  const formData = new FormData();
  formData.append("file", file);
  return fetch(`/visualize?format=${format}`, { method: "POST", body: formData });
}

function describeInfoForAlt(fileName, info) {
  const regionPart = info.side ? `${info.region} (${info.side})` : info.region;
  return `Визуализация ${fileName}: регион ${regionPart}, ориентиры и результаты чекеров — см. список под картинкой`;
}

const CHECKER_STATUS_LABELS = { ok: "оценено", not_evaluated: "не оценено" };

function renderVizInfo(info) {
  vizInfo.innerHTML = "";
  const regionRow = document.createElement("div");
  regionRow.className = "viz-info-row";
  const regionLabel = info.side ? `${info.region} (${info.side})` : info.region;
  regionRow.innerHTML = `<dt>Регион</dt><dd>${escapeHtml(regionLabel)}</dd>`;
  vizInfo.appendChild(regionRow);

  for (const [name, result] of Object.entries(info.checkers || {})) {
    const row = document.createElement("div");
    row.className = "viz-info-row";
    const statusLabel = CHECKER_STATUS_LABELS[result.status] || result.status;
    const verdict =
      result.status !== "ok"
        ? `${statusLabel}${result.reason ? ` — ${result.reason}` : ""}`
        : result.flag
          ? "нарушение"
          : "норма";
    const badgeClass = result.status === "ok" && result.flag ? "violation" : result.status === "ok" ? "ok" : "";
    const score = result.status === "ok" && result.score !== null ? ` (score=${Number(result.score).toFixed(3)})` : "";
    row.innerHTML = `<dt>${escapeHtml(name)}</dt><dd>${badgeClass ? `<span class="badge ${badgeClass}">${escapeHtml(verdict)}</span>` : escapeHtml(verdict)}${escapeHtml(score)}</dd>`;
    vizInfo.appendChild(row);
  }
}

function escapeHtml(value) {
  const div = document.createElement("div");
  div.textContent = value;
  return div.innerHTML;
}
