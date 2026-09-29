/**
 * Извлечение пикселей DICOM для отображения.
 *
 * Только рендеринг: классификация региона делается пайплайном columba
 * (см. bridge/describe.py), здесь пиксели нужны лишь чтобы нарисовать снимок.
 * Формат выгрузки известен: Implicit VR LE, MONOCHROME1/2, 8 или 16 бит.
 */

import { readFileSync } from "node:fs";
import dicomParser from "dicom-parser";

export interface PixelFrame {
  rows: number;
  cols: number;
  /** Яркости 0..255 (после нормализации и инверсии MONOCHROME1). */
  gray: Uint8Array;
}

export function readDicomPixels(path: string): PixelFrame {
  const bytes = new Uint8Array(readFileSync(path));
  const dataSet = dicomParser.parseDicom(bytes);

  const rows = dataSet.uint16("x00280010");
  const cols = dataSet.uint16("x00280011");
  const bitsAllocated = dataSet.uint16("x00280100") ?? 8;
  const photometric = (dataSet.string("x00280004") ?? "MONOCHROME2").trim();
  const pixelElement = dataSet.elements["x7fe00010"];
  if (rows === undefined || cols === undefined || !pixelElement) {
    throw new Error("нет пиксельных данных (Rows/Columns/PixelData)");
  }

  const count = rows * cols;
  let raw: Uint8Array | Uint16Array;
  if (bitsAllocated === 8) {
    raw = new Uint8Array(dataSet.byteArray.buffer, pixelElement.dataOffset, count);
  } else if (bitsAllocated === 16) {
    raw = new Uint16Array(count);
    const view = new DataView(dataSet.byteArray.buffer, pixelElement.dataOffset, count * 2);
    for (let i = 0; i < count; i += 1) raw[i] = view.getUint16(i * 2, true);
  } else {
    throw new Error(`неподдерживаемый BitsAllocated=${bitsAllocated}`);
  }

  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < count; i += 1) {
    const v = raw[i];
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const span = max > min ? max - min : 1;
  const invert = photometric === "MONOCHROME1";

  const gray = new Uint8Array(count);
  for (let i = 0; i < count; i += 1) {
    const norm = Math.round(((raw[i] - min) / span) * 255);
    gray[i] = invert ? 255 - norm : norm;
  }
  return { rows, cols, gray };
}
