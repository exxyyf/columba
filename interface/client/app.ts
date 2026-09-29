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
/** Активный фильтр по типу нарушения (строка словаря) или null — все. */
let activeViolation: string | null = null;
/** Свёрнута ли группа «без нарушений»; null — по умолчанию (зависит от размера). */
let cleanCollapsed: boolean | null = null;
/** Сколько «чистых» снимков показываем без сворачивания. */
const CLEAN_COLLAPSE_THRESHOLD = 12;
const pixelCache = new Map<string, ImageBitmap>();

const grid = document.getElementById("grid") as HTMLElement;
const summaryEl = document.getElementById("summary") as HTMLElement;
const warningsEl = document.getElementById("warnings") as HTMLElement;
const searchInput = document.getElementById("search") as HTMLInputElement;
const modal = document.getElementById("modal") as HTMLElement;
const modalCanvas = document.getElementById("modal-canvas") as HTMLCanvasElement;
const modalMeta = document.getElementById("modal-meta") as HTMLElement;
const modalOverlay = document.getElementById("modal-overlay") as HTMLImageElement;
const overlayToggle = document.getElementById("overlay-toggle") as HTMLButtonElement;
const downloadCsvBtn = document.getElementById("download-csv") as HTMLButtonElement;

/** Снимок, открытый в модалке (защита от гонок асинхронных дозагрузок). */
let modalFileId: string | null = null;
/** id -> objectURL PNG-оверлея из /api/visualize (кэш на сессию). */
const overlayCache = new Map<string, string>();

interface VizChecker {
  status: string;
  flag: boolean | null;
  score: number | null;
  reason: string | null;
}

interface VizInfo {
  region: string;
  side: string | null;
  checkers: Record<string, VizChecker>;
}

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
  renderViolationFilters();
  downloadCsvBtn.classList.toggle(
    "hidden",
    !(quality.state === "ready" && Object.keys(quality.results ?? {}).length > 0),
  );
  if (quality.state === "ready") {
    // Появились итоговые результаты: сетка перегруппировывается
    // (нарушения наверх, «чистые» — в сворачиваемую группу).
    renderGrid();
  } else {
    updateQualityBadges();
  }
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
// Фильтр по типам нарушений (строки словаря организаторов)
// ---------------------------------------------------------------------------

function violationTypes(row: QualityRow): string[] {
  return row.violation_type
    .split(";")
    .map((part) => part.trim())
    .filter((part) => part.length > 0);
}

function violationCounts(): Map<string, number> {
  const counts = new Map<string, number>();
  if (!quality.results) return counts;
  for (const row of Object.values(quality.results)) {
    if (row.quality_class !== 1) continue;
    for (const type of violationTypes(row)) {
      counts.set(type, (counts.get(type) ?? 0) + 1);
    }
  }
  return counts;
}

function renderViolationFilters(): void {
  const container = document.getElementById("violation-filters");
  if (!container) return;
  const counts = quality.state === "ready" ? violationCounts() : new Map<string, number>();
  if (counts.size === 0) {
    container.classList.add("hidden");
    if (activeViolation !== null) {
      activeViolation = null;
      renderGrid();
    }
    return;
  }
  if (activeViolation !== null && !counts.has(activeViolation)) {
    activeViolation = null;
  }
  container.classList.remove("hidden");
  const chips = [...counts.entries()].sort((a, b) => b[1] - a[1]);
  container.innerHTML = chips
    .map(([type, count]) => {
      const active = type === activeViolation;
      return `<button class="filter ${active ? "active" : ""}" type="button" data-violation="${escapeAttr(type)}" aria-pressed="${active}">${escapeHtml(type)}<span class="filter-count">${count}</span></button>`;
    })
    .join("");
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

function passesViolation(file: ViewerFile): boolean {
  if (activeViolation === null) return true;
  const row = qualityFor(file.id);
  if (!row || row.quality_class !== 1) return false;
  return violationTypes(row).includes(activeViolation);
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
  const files = manifest.files.filter((f) => passesFilter(f) && passesSearch(f) && passesViolation(f));
  if (files.length === 0) {
    grid.innerHTML = `<div class="loading">Ничего не найдено</div>`;
    return;
  }
  grid.innerHTML = "";

  // Пока контроль качества не досчитан — плоская сетка, как раньше.
  if (quality.state !== "ready" || !quality.results) {
    files.forEach(appendCard);
    return;
  }

  const violated = files.filter((f) => qualityFor(f.id)?.quality_class === 1);
  const clean = files.filter((f) => qualityFor(f.id)?.quality_class === 0);
  const unchecked = files.filter((f) => !qualityFor(f.id));

  // Группировка нужна, только когда есть что отделять: без нарушений
  // в текущей выборке сетка остаётся плоской.
  if (violated.length === 0 || clean.length + unchecked.length === 0) {
    files.forEach(appendCard);
    return;
  }

  appendGroupHeader(`✕ С нарушениями`, violated.length, "group-violated");
  violated.forEach(appendCard);

  if (clean.length > 0) {
    const collapsed = cleanCollapsed ?? clean.length > CLEAN_COLLAPSE_THRESHOLD;
    appendCleanHeader(clean.length, collapsed);
    if (!collapsed) clean.forEach(appendCard);
  }
  if (unchecked.length > 0) {
    appendGroupHeader("Без результата проверки", unchecked.length, "");
    unchecked.forEach(appendCard);
  }
}

function appendCard(file: ViewerFile): void {
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

function appendGroupHeader(title: string, count: number, extraClass: string): void {
  const header = document.createElement("div");
  header.className = `group-header ${extraClass}`.trim();
  header.innerHTML = `<span class="group-title">${escapeHtml(title)}</span><span class="group-count">${count}</span>`;
  grid.appendChild(header);
}

function appendCleanHeader(count: number, collapsed: boolean): void {
  const header = document.createElement("button");
  header.type = "button";
  header.className = "group-header group-clean group-toggle";
  header.setAttribute("aria-expanded", String(!collapsed));
  header.innerHTML = `
    <span class="group-title">✓ Без нарушений</span>
    <span class="group-count">${count}</span>
    <span class="group-action">${collapsed ? "Показать ▾" : "Свернуть ▴"}</span>`;
  header.addEventListener("click", () => {
    cleanCollapsed = !collapsed;
    renderGrid();
  });
  grid.appendChild(header);
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
  modalFileId = file.id;
  hideOverlay();
  overlayToggle.disabled = false;
  overlayToggle.classList.toggle("hidden", file.read_status !== "Success");

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
    <dl>${rows.map(([k, v]) => `<dt>${escapeHtml(k)}</dt><dd>${escapeHtml(v)}</dd>`).join("")}</dl>
    <div id="modal-checkers" class="modal-checkers"><p class="modal-label">Проверки</p><div class="checkers-loading">загружаются…</div></div>`;

  fetchBitmap(file.id)
    .then((bitmap) => drawToCanvas(modalCanvas, bitmap))
    .catch(() => undefined);

  if (file.read_status === "Success") {
    void loadCheckers(file.id);
  } else {
    const box = modalMeta.querySelector("#modal-checkers");
    if (box) box.innerHTML = "";
  }
}

// ---------------------------------------------------------------------------
// Разбор снимка: результаты чекеров и оверлей ориентиров из /visualize
// ---------------------------------------------------------------------------

const CHECKER_STATUS_RU: Record<string, string> = { not_evaluated: "не оценено" };

async function loadCheckers(id: string): Promise<void> {
  let html: string;
  try {
    const response = await fetch(`/api/visualize/${encodeURIComponent(id)}?format=json`);
    if (!response.ok) {
      const body = await response.json().catch(() => ({ error: response.statusText }));
      throw new Error(String(body.error ?? response.status));
    }
    const info = (await response.json()) as VizInfo;
    const rows = Object.entries(info.checkers ?? {}).map(([name, checker]) => {
      let verdict: string;
      let cls = "";
      if (checker.status !== "ok") {
        verdict = CHECKER_STATUS_RU[checker.status] ?? checker.status;
        if (checker.reason) verdict += ` — ${checker.reason}`;
      } else if (checker.flag) {
        verdict = "нарушение";
        cls = "checker-flag";
      } else {
        verdict = "норма";
        cls = "checker-ok";
      }
      const score =
        checker.status === "ok" && checker.score !== null && Number.isFinite(checker.score)
          ? ` <span class="checker-score">score ${Number(checker.score).toFixed(3)}</span>`
          : "";
      return `<dt>${escapeHtml(name)}</dt><dd class="${cls}">${escapeHtml(verdict)}${score}</dd>`;
    });
    html = rows.length > 0 ? `<p class="modal-label">Проверки</p><dl>${rows.join("")}</dl>` : "";
  } catch (error) {
    html = `<p class="modal-label">Проверки</p><div class="checkers-loading">недоступны: ${escapeHtml(String(error instanceof Error ? error.message : error))}</div>`;
  }
  if (modalFileId !== id) return; // за время запроса открыли другой снимок
  const box = modalMeta.querySelector("#modal-checkers");
  if (box) box.innerHTML = html;
}

function hideOverlay(): void {
  modalOverlay.classList.add("hidden");
  modalOverlay.removeAttribute("src");
  modalCanvas.classList.remove("hidden");
  overlayToggle.textContent = "Показать разметку";
}

overlayToggle.addEventListener("click", () => {
  if (!modalOverlay.classList.contains("hidden")) {
    hideOverlay();
    return;
  }
  const id = modalFileId;
  if (!id) return;
  void (async () => {
    overlayToggle.disabled = true;
    try {
      let url = overlayCache.get(id);
      if (!url) {
        const response = await fetch(`/api/visualize/${encodeURIComponent(id)}?format=png`);
        if (!response.ok) throw new Error(`оверлей не загрузился (${response.status})`);
        url = URL.createObjectURL(await response.blob());
        overlayCache.set(id, url);
      }
      if (modalFileId !== id) return;
      modalOverlay.src = url;
      modalOverlay.classList.remove("hidden");
      modalCanvas.classList.add("hidden");
      overlayToggle.textContent = "Скрыть разметку";
    } catch {
      overlayToggle.textContent = "Разметка недоступна";
    } finally {
      overlayToggle.disabled = false;
    }
  })();
});

function closeModal(): void {
  modal.classList.add("hidden");
  document.body.style.overflow = "";
  modalFileId = null;
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
  cleanCollapsed = null;
  renderGrid();
});

document.getElementById("violation-filters")?.addEventListener("click", (event) => {
  const target = (event.target as HTMLElement).closest<HTMLElement>(".filter");
  if (!target) return;
  const type = target.dataset.violation ?? null;
  activeViolation = activeViolation === type ? null : type;
  renderViolationFilters();
  renderGrid();
});

searchInput.addEventListener("input", () => {
  searchQuery = searchInput.value.trim().toLowerCase();
  renderGrid();
});

document.getElementById("rescan")?.addEventListener("click", () => void loadManifest(true));

// ---------------------------------------------------------------------------
// Скачивание результатов контроля качества (те же колонки, что сабмит)
// ---------------------------------------------------------------------------

function csvField(value: unknown): string {
  const text = value === null || value === undefined ? "" : String(value);
  return /["\n,;]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

downloadCsvBtn.addEventListener("click", () => {
  if (!manifest || !quality.results) return;
  const lines = ["file_name,anatomical_region,quality_class,quality_prob,violation_type"];
  for (const file of manifest.files) {
    const row = qualityFor(file.id);
    if (!row) continue;
    lines.push(
      [
        csvField(file.relative_path),
        csvField(row.anatomical_region),
        csvField(row.quality_class),
        csvField(row.quality_prob),
        csvField(row.violation_type),
      ].join(","),
    );
  }
  const blob = new Blob(["﻿" + lines.join("\r\n")], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "columba_results.csv";
  link.click();
  URL.revokeObjectURL(url);
});

// ---------------------------------------------------------------------------
// Загрузка собственных снимков: отдельные файлы, папка (структура
// исследований сохраняется — работает парное определение стороны бедра)
// или ZIP-архив (распаковывается на сервере). Партия уходит в одну папку,
// после последнего файла — пересканирование (та же классификация пайплайна).
// ---------------------------------------------------------------------------

const uploadButton = document.getElementById("upload") as HTMLButtonElement | null;
const uploadMenu = document.getElementById("upload-menu") as HTMLElement | null;
const uploadInput = document.getElementById("upload-input") as HTMLInputElement | null;
const uploadFolderInput = document.getElementById("upload-folder-input") as HTMLInputElement | null;
const uploadZipInput = document.getElementById("upload-zip-input") as HTMLInputElement | null;

function toggleUploadMenu(open?: boolean): void {
  if (!uploadMenu || !uploadButton) return;
  const show = open ?? uploadMenu.classList.contains("hidden");
  uploadMenu.classList.toggle("hidden", !show);
  uploadButton.setAttribute("aria-expanded", String(show));
}

uploadButton?.addEventListener("click", (event) => {
  event.stopPropagation();
  toggleUploadMenu();
});

uploadMenu?.addEventListener("click", (event) => {
  const item = (event.target as HTMLElement).closest<HTMLElement>("[data-upload]");
  if (!item) return;
  toggleUploadMenu(false);
  if (item.dataset.upload === "files") uploadInput?.click();
  else if (item.dataset.upload === "folder") uploadFolderInput?.click();
  else if (item.dataset.upload === "zip") uploadZipInput?.click();
});

document.addEventListener("click", (event) => {
  if (uploadMenu && !uploadMenu.classList.contains("hidden")) {
    const inside = (event.target as HTMLElement).closest(".upload-menu-wrap");
    if (!inside) toggleUploadMenu(false);
  }
});

uploadInput?.addEventListener("change", () => {
  const files = Array.from(uploadInput.files ?? []);
  uploadInput.value = "";
  if (files.length === 0 || !uploadButton) return;
  void uploadFiles(
    files.map((file) => ({ file, rel: file.name })),
    uploadButton,
  );
});

uploadFolderInput?.addEventListener("change", () => {
  const all = Array.from(uploadFolderInput.files ?? []);
  uploadFolderInput.value = "";
  if (!uploadButton) return;
  // `webkitRelativePath` включает имя ВЫБРАННОЙ папки первым сегментом —
  // отрезаем его (то же соглашение, что у /predict/zip: имя корня не входит
  // в relative_path), берём только .dcm.
  const entries = all
    .filter((file) => /\.dcm$/i.test(file.name))
    .map((file) => {
      const parts = file.webkitRelativePath.split("/");
      return { file, rel: parts.length > 1 ? parts.slice(1).join("/") : file.name };
    });
  if (entries.length === 0) {
    alert("В выбранной папке нет .dcm-файлов");
    return;
  }
  void uploadFiles(entries, uploadButton);
});

uploadZipInput?.addEventListener("change", () => {
  const file = uploadZipInput.files?.[0] ?? null;
  uploadZipInput.value = "";
  if (!file || !uploadButton) return;
  void uploadZip(file, uploadButton);
});

function newBatchLabel(): string {
  return new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");
}

async function uploadFiles(
  entries: { file: File; rel: string }[],
  button: HTMLButtonElement,
): Promise<void> {
  const label = button.innerHTML;
  button.disabled = true;
  const batch = newBatchLabel();
  const failures: string[] = [];
  for (let i = 0; i < entries.length; i += 1) {
    button.innerHTML = `Загрузка ${i + 1} из ${entries.length}…`;
    try {
      const response = await fetch("/api/upload", {
        method: "POST",
        body: entries[i].file,
        headers: {
          "Content-Type": "application/octet-stream",
          "X-File-Name": encodeURIComponent(entries[i].file.name),
          "X-Relative-Path": encodeURIComponent(entries[i].rel),
          "X-Upload-Batch": batch,
        },
      });
      if (!response.ok) {
        const body = await response.json().catch(() => ({ error: response.statusText }));
        failures.push(`${entries[i].rel}: ${String(body.error ?? response.statusText)}`);
      }
    } catch (error) {
      failures.push(`${entries[i].rel}: ${String(error instanceof Error ? error.message : error)}`);
    }
  }
  button.innerHTML = label;
  button.disabled = false;
  if (failures.length > 0) {
    alert(`Не удалось загрузить:\n${failures.join("\n")}`);
  }
  if (failures.length < entries.length) {
    await loadManifest(true);
  }
}

async function uploadZip(file: File, button: HTMLButtonElement): Promise<void> {
  const label = button.innerHTML;
  button.disabled = true;
  button.innerHTML = "Загрузка архива…";
  try {
    const response = await fetch("/api/upload/zip", {
      method: "POST",
      body: file,
      headers: {
        "Content-Type": "application/octet-stream",
        "X-Upload-Batch": newBatchLabel(),
      },
    });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) {
      alert(`Не удалось загрузить архив: ${String(body.error ?? response.statusText)}`);
      return;
    }
    const skipped = Array.isArray(body.skipped) ? body.skipped.length : 0;
    if (skipped > 0) {
      alert(`Пропущено записей с некорректными путями: ${skipped}`);
    }
    await loadManifest(true);
  } catch (error) {
    alert(`Не удалось загрузить архив: ${String(error instanceof Error ? error.message : error)}`);
  } finally {
    button.innerHTML = label;
    button.disabled = false;
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
