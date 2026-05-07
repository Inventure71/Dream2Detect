import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import sharp from "sharp";
import { parseCsv } from "./csv-utils.mjs";

const appRoot = process.cwd();
const repoRoot = process.env.DREAM2DETECT_REPO_ROOT ?? path.resolve(appRoot, "../..");
const reviewedCsvPath =
  process.env.SYNTHETIC_REVIEWED_CSV_PATH ??
  path.join(repoRoot, "data/datasets/synthetic_reviewed_seed_set_phase1_round1.csv");
const startingCsvPath =
  process.env.SYNTHETIC_STARTING_CSV_PATH ??
  path.join(repoRoot, "data/datasets/synthetic_starting_dataset_phase1_round1.csv");
const outputDir = path.join(appRoot, "public/band-examples");

const bands = [
  "0-10",
  "11-20",
  "21-30",
  "31-35",
  "36-45",
  "46-55",
  "56-65",
  "66-75",
  "76-85",
  "86-100",
];

function publicNameForBand(band) {
  return `${band.replace("-", "_")}.jpg`;
}

function selectExamples(reviewedRows, startingRows) {
  const selected = new Map();

  for (const row of reviewedRows) {
    if (row.qc_status !== "accepted_as_labeled") continue;
    if (row.training_score_band && !selected.has(row.training_score_band)) {
      selected.set(row.training_score_band, { ...row, example_source: "reviewed" });
    }
  }

  for (const row of startingRows) {
    if (!row.training_score_band || selected.has(row.training_score_band)) continue;
    selected.set(row.training_score_band, { ...row, example_source: "intended_unreviewed" });
  }

  return selected;
}

async function main() {
  const reviewedRows = parseCsv(await readFile(reviewedCsvPath, "utf8"));
  const startingRows = parseCsv(await readFile(startingCsvPath, "utf8"));
  const selected = selectExamples(reviewedRows, startingRows);

  await mkdir(outputDir, { recursive: true });

  const manifest = [];
  for (const band of bands) {
    const row = selected.get(band);
    if (!row) {
      throw new Error(`No band example source found for ${band}`);
    }

    const inputPath = path.resolve(row.image_path);
    const outputName = publicNameForBand(band);
    const outputPath = path.join(outputDir, outputName);

    await sharp(inputPath)
      .resize({ width: 360, height: 240, fit: "cover", position: "attention" })
      .jpeg({ quality: 76, mozjpeg: true })
      .toFile(outputPath);

    manifest.push({
      band,
      image: `/band-examples/${outputName}`,
      title: row.prompt_title,
      source: row.example_source,
    });
  }

  await writeFile(
    path.join(outputDir, "manifest.json"),
    JSON.stringify(manifest, null, 2) + "\n",
    "utf8",
  );

  for (const item of manifest) {
    console.log(`${item.band}: ${item.image} (${item.source})`);
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
