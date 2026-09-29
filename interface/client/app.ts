/**
 * Клиент просмотрщика DXA.
 *
 * Показывает снимки из каталогов данных; регион («Поясничный отдел
 * позвоночника» / «Проксимальный отдел бедра») и сторону бедра определяет
 * пайплайн columba на сервере — здесь только отрисовка и фильтры.
 *
 * Пиксель анизотропный (1,05 мм по Y, 0,6 мм по X), поэтому канвас
 * растягивается до физических пропорций через CSS aspect-ratio.
 */

interface ViewerFile {
  id: string;
  file_name: string;
  relative_path: string;
  study_folder: string;
  root: string;
  read_status: string;
  read_error: string | null;
  rows: number | null;
  cols: number | null;
  region: string;
  region_final: string;
  hip_side_final: string | null;
  hip_side_score: number | null;
  hip_side_confident: boolean;
  hip_side_method: string | null;
  region_method: string;
  region_disagreement: boolean;
  cnn_label: string | null;
  dedup_group_size: number | null;
}

interface ViewerManifest {
  generated_at: string;
  cnn_available: boolean;
  region_labels: Record<string, string>;
  pixel_spacing_mm: { y: number; x: number };
  roots: string[];
  files: ViewerFile[];
  warnings: string[];
}

const SIDE_RU: Record<string, string> = { left: "левое", right: "правое" };
const METHOD_RU: Record<string, string> = {
  heuristic_primary: "эвристика (CNN подтвердила)",
  heuristic: "эвристика",
  cnn: "CNN",
  cnn_primary: "CNN",
};

interface QualityRow {
  anatomical_region: string;
  quality_class: number;
  quality_prob: number;
  violation_type: string;
}

interface QualityResponse {
  state: "idle" | "running" | "ready" | "error";
  done?: number;
  total?: number;
  message?: string;
  results?: Record<string, QualityRow>;
}

let manifest: ViewerManifest | null = null;
let activeFilter = "all";
let searchQuery = "";
let quality: QualityResponse = { state: "idle" };
let qualityTimer: number | null = null;
const pixelCache = new Map<string, ImageBitmap>();

const grid = document.getElementById("grid") as HTMLElement;
const summaryEl = document.getElementById("summary") as HTMLElement;
const warningsEl = document.getElementById("warnings") as HTMLElement;
const searchInput = document.getElementById("search") as HTMLInputElement;
const modal = document.getElementById("modal") as HTMLElement;
const modalCanvas = document.getElementById("modal-canvas") as HTMLCanvasElement;
const modalMeta = document.getElementById("modal-meta") as HTMLElement;

// ---------------------------------------------------------------------------
// Загрузка манифеста
// ---------------------------------------------------------------------------

async function loadManifest(force = false): Promise<void> {
  grid.innerHTML = `<div class="loading">Загрузка и классификация снимков…</div>`;
  const response = await fetch(`/api/manifest${force ? "?force=1" : ""}`);
  if (!response.ok) {
    const body = await response.json().catch(() => ({ error: response.statusText }));
    grid.innerHTML = `<div class="loading">Ошибка: ${escapeHtml(String(body.error ?? "не удалось загрузить"))}</div>`;
    return;
  }
  manifest = (await response.json()) as ViewerManifest;
  renderSummary();
  renderWarnings();
  renderGrid();
  void pollQuality();
}

// ---------------------------------------------------------------------------
// Контроль качества: опрос сервера, пока идёт расчёт по пайплайну
// ---------------------------------------------------------------------------

async function pollQuality(): Promise<void> {
  if (qualityTimer !== null) {
    window.clearTimeout(qualityTimer);
    qualityTimer = null;
  }
  try {
    const response = await fetch("/api/quality");
    quality = response.ok ? ((await response.json()) as QualityResponse) : { state: "error" };
  } catch {
    quality = { state: "error", message: "сервис качества недоступен" };
  }
  renderSummary();
  updateQualityBadges();
  if (quality.state === "running") {
    qualityTimer = window.setTimeout(() => void pollQuality(), 2500);
  }
}

function qualityFor(id: string): QualityRow | null {
  return quality.results?.[id] ?? null;
}

function qualityBadge(id: string): string {
  const row = qualityFor(id);
  if (!row) {
    return quality.state === "running" || quality.state === "idle"
      ? `<span class="badge badge-side" title="Контроль качества ещё выполняется">качество…</span>`
      : "";
  }
  if (row.quality_class === 1) {
    const types = row.violation_type || "нарушение";
    return `<span class="badge badge-error" title="${escapeAttr(`Вероятность нарушения ${row.quality_prob.toFixed(2)}`)}">✕ ${escapeHtml(types)}</span>`;
  }
  return `<span class="badge badge-ok" title="${escapeAttr(`Вероятность нарушения ${row.quality_prob.toFixed(2)}`)}">✓ без нарушений</span>`;
}

function updateQualityBadges(): void {
  document.querySelectorAll<HTMLElement>("[data-quality-for]").forEach((slot) => {
    slot.innerHTML = qualityBadge(slot.dataset.qualityFor ?? "");
  });
}

// ---------------------------------------------------------------------------
// Сводка и предупреждения
// ---------------------------------------------------------------------------

function renderSummary(): void {
  if (!manifest) return;
  const files = manifest.files;
  const spine = files.filter((f) => f.region_final === "spine").length;
  const hipLeft = files.filter((f) => f.region_final === "hip" && f.hip_side_final === "left").length;
  const hipRight = files.filter((f) => f.region_final === "hip" && f.hip_side_final === "right").length;
  const hipNoSide = files.filter((f) => f.region_final === "hip" && !f.hip_side_final).length;
  const failures = files.filter((f) => f.read_status !== "Success").length;
  const disagreements = files.filter((f) => f.region_disagreement).length;

  const hipDetail = `левое ${hipLeft} · правое ${hipRight}${hipNoSide ? ` · без стороны ${hipNoSide}` : ""}`;
  summaryEl.innerHTML = `
    <div class="summary-item">
      <strong class="summary-value">${files.length}</strong>
      <span class="summary-label">Всего снимков</span>
    </div>
    <div class="summary-item">
      <strong class="summary-value">${spine}</strong>
      <span class="summary-label">Позвоночник</span>
    </div>
    <div class="summary-item">
      <strong class="summary-value">${hipLeft + hipRight + hipNoSide}</strong>
      <span class="summary-label">Бедро</span>
      <span class="summary-detail">${hipDetail}</span>
    </div>
    <div class="summary-item">
      <strong class="summary-value summary-value-text">${manifest.cnn_available ? "CNN активна" : "Только эвристика"}</strong>
      <span class="summary-label">Режим классификации</span>
    </div>
    ${
      disagreements
        ? `<div class="summary-item summary-item-attention"><strong class="summary-value">${disagreements}</strong><span class="summary-label">Расхождения веток</span></div>`
        : ""
    }
    ${
      failures
        ? `<div class="summary-item summary-item-error"><strong class="summary-value">${failures}</strong><span class="summary-label">Ошибки чтения</span></div>`
        : ""
    }
    ${qualitySummary()}`;
}

function qualitySummary(): string {
  if (quality.state === "running") {
    const done = quality.done ?? 0;
    const total = quality.total ?? 0;
    return `<div class="summary-item"><strong class="summary-value summary-value-text">${done} / ${total}</strong><span class="summary-label">Контроль качества…</span></div>`;
  }
  if (quality.state === "error") {
    return `<div class="summary-item summary-item-error"><strong class="summary-value summary-value-text">недоступен</strong><span class="summary-label" title="${escapeAttr(quality.message ?? "")}">Контроль качества</span></div>`;
  }
  if (quality.state === "ready" && quality.results) {
    const bad = Object.values(quality.results).filter((r) => r.quality_class === 1).length;
    return `<div class="summary-item ${bad ? "summary-item-attention" : ""}"><strong class="summary-value">${bad}</strong><span class="summary-label">С нарушениями</span></div>`;
  }
  return "";
}

function renderWarnings(): void {
  if (!manifest || manifest.warnings.length === 0) {
    warningsEl.classList.add("hidden");
    return;
  }
  warningsEl.classList.remove("hidden");
  warningsEl.textContent = manifest.warnings.join("\n");
}

// ---------------------------------------------------------------------------
// Сетка карточек
// ---------------------------------------------------------------------------

function needsAttention(file: ViewerFile): boolean {
  return (
    file.read_status !== "Success" ||
    file.region_disagreement ||
    file.region_final === "unknown" ||
    (file.region_final === "hip" && !file.hip_side_final)
  );
}

function passesFilter(file: ViewerFile): boolean {
  switch (activeFilter) {
    case "spine":
      return file.region_final === "spine";
    case "hip-left":
      return file.region_final === "hip" && file.hip_side_final === "left";
    case "hip-right":
      return file.region_final === "hip" && file.hip_side_final === "right";
    case "attention":
      return needsAttention(file);
    case "violations":
      return qualityFor(file.id)?.quality_class === 1;
    default:
      return true;
  }
}

function passesSearch(file: ViewerFile): boolean {
  if (!searchQuery) return true;
  const haystack = `${file.file_name} ${file.relative_path} ${file.study_folder}`.toLowerCase();
  return haystack.includes(searchQuery);
}

const observer = new IntersectionObserver(
  (entries) => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      const card = entry.target as HTMLElement;
      observer.unobserve(card);
      const id = card.dataset.fileId;
      if (id) void drawThumb(id, card);
    }
  },
  { rootMargin: "300px" },
);

function renderGrid(): void {
  if (!manifest) return;
  observer.disconnect();
  const files = manifest.files.filter((f) => passesFilter(f) && passesSearch(f));
  if (files.length === 0) {
    grid.innerHTML = `<div class="loading">Ничего не найдено</div>`;
    return;
  }
  grid.innerHTML = "";
  for (const file of files) {
    const card = document.createElement("button");
    card.className = "card";
    card.type = "button";
    card.dataset.fileId = file.id;
    card.setAttribute("aria-label", `Открыть снимок ${file.file_name}`);
    card.innerHTML = `
      <div class="thumb"><canvas></canvas></div>
      <div class="card-info">
        <div class="card-title" title="${escapeAttr(file.file_name)}">${escapeHtml(file.file_name)}</div>
        <div class="card-path" title="${escapeAttr(file.relative_path)}">${escapeHtml(shortStudy(file))}</div>
        <div class="badges">${badges(file)}<span data-quality-for="${escapeAttr(file.id)}">${qualityBadge(file.id)}</span></div>
      </div>`;
    card.addEventListener("click", () => openModal(file));
    grid.appendChild(card);
    observer.observe(card);
  }
}

function shortStudy(file: ViewerFile): string {
  return file.study_folder === "." ? file.relative_path : file.study_folder;
}

function badges(file: ViewerFile): string {
  if (!manifest) return "";
  const parts: string[] = [];
  if (file.read_status !== "Success") {
    parts.push(`<span class="badge badge-error">ошибка чтения</span>`);
    return parts.join("");
  }
  const label = manifest.region_labels[file.region_final];
  if (file.region_final === "spine") {
    parts.push(`<span class="badge badge-spine" title="${escapeAttr(label ?? "")}">Позвоночник</span>`);
  } else if (file.region_final === "hip") {
    parts.push(`<span class="badge badge-hip" title="${escapeAttr(label ?? "")}">Бедро</span>`);
    const side = file.hip_side_final ? SIDE_RU[file.hip_side_final] ?? file.hip_side_final : "сторона?";
    parts.push(`<span class="badge badge-side">${escapeHtml(side)}</span>`);
  } else {
    parts.push(`<span class="badge badge-warn">регион не определён</span>`);
  }
  if (file.region_disagreement) {
    parts.push(`<span class="badge badge-warn" title="Эвристика и CNN разошлись">⚠ CNN ≠ эвристика</span>`);
  }
  return parts.join("");
}

// ---------------------------------------------------------------------------
// Отрисовка пикселей
// ---------------------------------------------------------------------------

async function fetchBitmap(id: string): Promise<ImageBitmap> {
  const cached = pixelCache.get(id);
  if (cached) return cached;
  const response = await fetch(`/api/pixels/${encodeURIComponent(id)}`);
  if (!response.ok) throw new Error(`пиксели не загрузились (${response.status})`);
  const rows = Number(response.headers.get("X-Rows"));
  const cols = Number(response.headers.get("X-Cols"));
  const gray = new Uint8Array(await response.arrayBuffer());
  const rgba = new Uint8ClampedArray(rows * cols * 4);
  for (let i = 0; i < gray.length; i += 1) {
    const v = gray[i];
    rgba[i * 4] = v;
    rgba[i * 4 + 1] = v;
    rgba[i * 4 + 2] = v;
    rgba[i * 4 + 3] = 255;
  }
  const bitmap = await createImageBitmap(new ImageData(rgba, cols, rows));
  pixelCache.set(id, bitmap);
  return bitmap;
}

/** Нарисовать снимок с поправкой на анизотропию пикселя. */
function drawToCanvas(canvas: HTMLCanvasElement, bitmap: ImageBitmap): void {
  if (!manifest) return;
  const spacing = manifest.pixel_spacing_mm;
  // Физические пропорции: высота пикселя 1,05 мм против 0,6 мм по ширине.
  const stretchY = spacing.y / spacing.x;
  canvas.width = bitmap.width;
  canvas.height = Math.round(bitmap.height * stretchY);
  const context = canvas.getContext("2d");
  if (!context) return;
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";
  context.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
}

async function drawThumb(id: string, card: HTMLElement): Promise<void> {
  const canvas = card.querySelector("canvas");
  if (!canvas) return;
  try {
    const bitmap = await fetchBitmap(id);
    drawToCanvas(canvas, bitmap);
  } catch {
    const thumb = card.querySelector(".thumb");
    if (thumb) thumb.innerHTML = `<span class="badge badge-error">нет изображения</span>`;
  }
}

// ---------------------------------------------------------------------------
// Модальное окно
// ---------------------------------------------------------------------------

function openModal(file: ViewerFile): void {
  if (!manifest) return;
  modal.classList.remove("hidden");
  document.body.style.overflow = "hidden";
  modalCanvas.width = 1;
  modalCanvas.height = 1;

  const rows: [string, string][] = [
    ["Файл", file.file_name],
    ["Путь", file.relative_path],
    ["Исследование", file.study_folder === "." ? "— (плоский каталог)" : file.study_folder],
    ["Статус чтения", file.read_status + (file.read_error ? ` — ${file.read_error}` : "")],
    ["Размер кадра", file.rows && file.cols ? `${file.cols} × ${file.rows} px` : "—"],
    ["Регион", manifest.region_labels[file.region_final] ?? file.region_final],
    ["Метод", METHOD_RU[file.region_method] ?? file.region_method],
    ["Ответ CNN", file.cnn_label ?? "—"],
  ];
  if (file.region_final === "hip") {
    const side = file.hip_side_final ? SIDE_RU[file.hip_side_final] ?? file.hip_side_final : "не определена";
    rows.push(["Сторона бедра", side]);
    rows.push([
      "Признак стороны",
      file.hip_side_score !== null
        ? `${file.hip_side_score.toFixed(3)}${file.hip_side_confident ? "" : " (неуверенно)"}`
        : "—",
    ]);
    if (file.hip_side_method) rows.push(["Метод стороны", file.hip_side_method]);
  }
  if (file.region_disagreement) {
    rows.push(["Внимание", "эвристика и CNN разошлись — принята эвристика"]);
  }
  if (file.dedup_group_size && file.dedup_group_size > 1) {
    rows.push(["Дубликаты", `${file.dedup_group_size} файла(ов) с одинаковыми пикселями`]);
  }
  const qualityRow = qualityFor(file.id);
  if (qualityRow) {
    rows.push([
      "Контроль качества",
      qualityRow.quality_class === 1 ? "нарушение обнаружено" : "без нарушений",
    ]);
    rows.push(["Вероятность нарушения", qualityRow.quality_prob.toFixed(3)]);
    if (qualityRow.violation_type) rows.push(["Тип нарушения", qualityRow.violation_type]);
  } else if (quality.state === "running") {
    rows.push(["Контроль качества", "выполняется…"]);
  }

  modalMeta.innerHTML = `
    <p class="modal-label">Сведения о снимке</p>
    <h2 id="modal-title">${escapeHtml(file.file_name)}</h2>
    <div class="sub">${escapeHtml(file.root)}</div>
    <dl>${rows.map(([k, v]) => `<dt>${escapeHtml(k)}</dt><dd>${escapeHtml(v)}</dd>`).join("")}</dl>`;

  fetchBitmap(file.id)
    .then((bitmap) => drawToCanvas(modalCanvas, bitmap))
    .catch(() => undefined);
}

function closeModal(): void {
  modal.classList.add("hidden");
  document.body.style.overflow = "";
}

// ---------------------------------------------------------------------------
// Утилиты и обвязка событий
// ---------------------------------------------------------------------------

function escapeHtml(text: string): string {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function escapeAttr(text: string): string {
  return escapeHtml(text);
}

document.getElementById("filters")?.addEventListener("click", (event) => {
  const target = event.target as HTMLElement;
  if (!target.classList.contains("filter")) return;
  document.querySelectorAll(".filter").forEach((el) => {
    el.classList.remove("active");
    el.setAttribute("aria-pressed", "false");
  });
  target.classList.add("active");
  target.setAttribute("aria-pressed", "true");
  activeFilter = target.dataset.filter ?? "all";
  renderGrid();
});

searchInput.addEventListener("input", () => {
  searchQuery = searchInput.value.trim().toLowerCase();
  renderGrid();
});

document.getElementById("rescan")?.addEventListener("click", () => void loadManifest(true));

// ---------------------------------------------------------------------------
// Загрузка собственных файлов: партия уходит в одну папку-«исследование»,
// после последнего файла — пересканирование (та же классификация пайплайна).
// ---------------------------------------------------------------------------

const uploadButton = document.getElementById("upload") as HTMLButtonElement | null;
const uploadInput = document.getElementById("upload-input") as HTMLInputElement | null;

uploadButton?.addEventListener("click", () => uploadInput?.click());

uploadInput?.addEventListener("change", () => {
  const files = Array.from(uploadInput.files ?? []);
  uploadInput.value = "";
  if (files.length === 0 || !uploadButton) return;
  void uploadFiles(files, uploadButton);
});

async function uploadFiles(files: File[], button: HTMLButtonElement): Promise<void> {
  const label = button.innerHTML;
  button.disabled = true;
  const batch = new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");
  const failures: string[] = [];
  for (let i = 0; i < files.length; i += 1) {
    button.innerHTML = `Загрузка ${i + 1} из ${files.length}…`;
    try {
      const response = await fetch("/api/upload", {
        method: "POST",
        body: files[i],
        headers: {
          "Content-Type": "application/octet-stream",
          "X-File-Name": encodeURIComponent(files[i].name),
          "X-Upload-Batch": batch,
        },
      });
      if (!response.ok) {
        const body = await response.json().catch(() => ({ error: response.statusText }));
        failures.push(`${files[i].name}: ${String(body.error ?? response.statusText)}`);
      }
    } catch (error) {
      failures.push(`${files[i].name}: ${String(error instanceof Error ? error.message : error)}`);
    }
  }
  button.innerHTML = label;
  button.disabled = false;
  if (failures.length > 0) {
    alert(`Не удалось загрузить:\n${failures.join("\n")}`);
  }
  if (failures.length < files.length) {
    await loadManifest(true);
  }
}
document.getElementById("modal-close")?.addEventListener("click", closeModal);
modal.addEventListener("click", (event) => {
  if (event.target === modal) closeModal();
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") closeModal();
});

void loadManifest();
