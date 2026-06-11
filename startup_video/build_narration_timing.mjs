import fs from "node:fs";
import path from "node:path";

const ROOT = path.resolve(".");
const segmentDir = path.join(ROOT, "assets", "sapi_segments_rate2");
const outDir = path.join(ROOT, "assets");
const pauseDuration = 0.35;

function wavDuration(filePath) {
  const data = fs.readFileSync(filePath);
  let byteRate = 0;
  let dataSize = 0;
  for (let i = 12; i < data.length - 8;) {
    const id = data.toString("ascii", i, i + 4);
    const size = data.readUInt32LE(i + 4);
    if (id === "fmt ") byteRate = data.readUInt32LE(i + 16);
    if (id === "data") dataSize = size;
    i += 8 + size + (size % 2);
  }
  if (!byteRate || !dataSize) throw new Error(`Cannot read WAV duration: ${filePath}`);
  return dataSize / byteRate;
}

const segments = fs
  .readdirSync(segmentDir)
  .filter((name) => /^segment_\d+\.wav$/.test(name))
  .sort((a, b) => a.localeCompare(b));

let cursor = 0;
const timings = segments.map((name, index) => {
  const duration = wavDuration(path.join(segmentDir, name));
  const start = cursor;
  const end = start + duration;
  cursor = end + (index === segments.length - 1 ? 0 : pauseDuration);
  return {
    slide: index + 1,
    file: `assets/sapi_segments_rate2/${name}`,
    start: +start.toFixed(3),
    end: +end.toFixed(3),
    duration: +duration.toFixed(3),
  };
});

const concatLines = [];
for (let i = 0; i < segments.length; i++) {
  concatLines.push(`file 'sapi_segments_rate2/${segments[i]}'`);
  if (i !== segments.length - 1) concatLines.push("file 'pause_035.wav'");
}

fs.writeFileSync(path.join(outDir, "narration_timing.json"), JSON.stringify(timings, null, 2), "utf8");
fs.writeFileSync(path.join(outDir, "narration_concat.txt"), `${concatLines.join("\n")}\n`, "utf8");
console.log(JSON.stringify({ segments: segments.length, totalDuration: +cursor.toFixed(3) }, null, 2));
