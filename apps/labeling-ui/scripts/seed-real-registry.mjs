import { createClient } from "@supabase/supabase-js";
import { readFile } from "node:fs/promises";
import path from "node:path";
import sharp from "sharp";
import "./load-env.mjs";
import { parseCsv } from "./csv-utils.mjs";

const appRoot = process.cwd();
const repoRoot = process.env.DREAM2DETECT_REPO_ROOT ?? path.resolve(appRoot, "../..");
const registryPath =
  process.env.REAL_REGISTRY_PATH ??
  path.join(repoRoot, "data/registries/real_relabel_registry.csv");

const supabaseUrl = process.env.SUPABASE_URL;
const serviceRoleKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
const bucket = process.env.SUPABASE_STORAGE_BUCKET ?? "dream2detect-images";
const datasetSlug = process.env.SUPABASE_DATASET_SLUG ?? "redf0xwin_boxes_cardboard";
const datasetName =
  process.env.SUPABASE_DATASET_NAME ?? "Redf0xwin Boxes/Cardboard Defects";
const datasetKind = process.env.SUPABASE_DATASET_KIND ?? "real";
const maxDisplayDimension = Number(process.env.LABELING_MAX_IMAGE_DIMENSION ?? 1280);
const jpegQuality = Number(process.env.LABELING_JPEG_QUALITY ?? 72);
const cleanupOriginalUploads = process.env.SUPABASE_CLEANUP_ORIGINAL_UPLOADS === "true";
const uploadRetries = Number(process.env.SUPABASE_UPLOAD_RETRIES ?? 5);
const sourceUrl =
  process.env.SUPABASE_DATASET_SOURCE_URL ??
  "https://www.kaggle.com/datasets/redf0xwin/recognizing-defects-in-boxes-and-cardboard";

if (!supabaseUrl || !serviceRoleKey) {
  throw new Error("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required.");
}

const supabase = createClient(supabaseUrl, serviceRoleKey, {
  auth: { persistSession: false },
});

async function createDisplayImage(fileBuffer) {
  return sharp(fileBuffer)
    .rotate()
    .resize({
      width: maxDisplayDimension,
      height: maxDisplayDimension,
      fit: "inside",
      withoutEnlargement: true,
    })
    .jpeg({
      quality: jpegQuality,
      mozjpeg: true,
    })
    .toBuffer();
}

function wait(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function withRetry(label, operation) {
  let lastError;
  for (let attempt = 1; attempt <= uploadRetries; attempt += 1) {
    try {
      return await operation();
    } catch (error) {
      lastError = error;
      const retryableStatus = Number(error?.statusCode ?? error?.status ?? 0);
      const retryable =
        retryableStatus === 0 ||
        retryableStatus === 429 ||
        (retryableStatus >= 500 && retryableStatus < 600);

      if (!retryable || attempt === uploadRetries) {
        throw error;
      }

      const delayMs = 750 * attempt;
      console.warn(`${label} failed on attempt ${attempt}; retrying in ${delayMs}ms`);
      await wait(delayMs);
    }
  }

  throw lastError;
}

async function main() {
  const csvText = await readFile(registryPath, "utf8");
  const registryRows = parseCsv(csvText);

  const { data: buckets, error: bucketsError } = await supabase.storage.listBuckets();
  if (bucketsError) throw bucketsError;

  if (!buckets.some((item) => item.name === bucket)) {
    const { error: createBucketError } = await supabase.storage.createBucket(bucket, {
      public: true,
      fileSizeLimit: "20MB",
      allowedMimeTypes: ["image/jpeg", "image/png", "image/webp"],
    });
    if (createBucketError) throw createBucketError;
  }

  const { data: dataset, error: datasetError } = await supabase
    .from("labeling_datasets")
    .upsert(
      {
        slug: datasetSlug,
        name: datasetName,
        kind: datasetKind,
        source_url: sourceUrl,
        storage_bucket: bucket,
        storage_prefix: datasetSlug,
      },
      { onConflict: "slug" },
    )
    .select("id")
    .single();

  if (datasetError) throw datasetError;

  const { data: existingImages, error: existingError } = await supabase
    .from("labeling_images")
    .select("image_id, storage_path")
    .eq("dataset_id", dataset.id);

  if (existingError) throw existingError;

  const existingIds = new Set(existingImages.map((item) => item.image_id));
  const existingStoragePaths = new Map(
    existingImages.map((item) => [item.image_id, item.storage_path]),
  );
  const rowsToInsert = [];
  let uploaded = 0;
  let alreadyDisplay = 0;
  let originalUploadsRemoved = 0;
  let skippedExisting = 0;

  for (const row of registryRows) {
    const localImagePath = row.image_path;
    if (!localImagePath) continue;

    const absoluteImagePath = path.resolve(repoRoot, localImagePath);
    const storagePath = `${datasetSlug}/display/${path.parse(localImagePath).name}.jpg`;
    const currentStoragePath = existingStoragePaths.get(row.image_id);

    if (currentStoragePath === storagePath) {
      alreadyDisplay += 1;
    } else {
      const fileBuffer = await readFile(absoluteImagePath);
      const displayBuffer = await createDisplayImage(fileBuffer);

      await withRetry(`upload ${storagePath}`, async () => {
        const { error: uploadError } = await supabase.storage
          .from(bucket)
          .upload(storagePath, displayBuffer, {
            contentType: "image/jpeg",
            upsert: true,
          });

        if (uploadError) throw uploadError;
      });

      uploaded += 1;
    }

    if (existingIds.has(row.image_id)) {
      await withRetry(`update ${row.image_id}`, async () => {
        const { error: updateError } = await supabase
          .from("labeling_images")
          .update({
            storage_bucket: bucket,
            storage_path: storagePath,
            local_image_path: localImagePath,
            source_dataset: row.source_dataset,
            original_label: row.original_label,
          })
          .eq("dataset_id", dataset.id)
          .eq("image_id", row.image_id);

        if (updateError) throw updateError;
      });

      if (cleanupOriginalUploads) {
        const oldRootPath = `${datasetSlug}/${path.basename(localImagePath)}`;
        await withRetry(`remove ${oldRootPath}`, async () => {
          const { error: removeError } = await supabase.storage.from(bucket).remove([oldRootPath]);
          if (removeError) throw removeError;
        }).then(
          () => {
            originalUploadsRemoved += 1;
          },
          () => undefined,
        );
      }

      skippedExisting += 1;
      continue;
    }

    rowsToInsert.push({
      dataset_id: dataset.id,
      image_id: row.image_id,
      source_image_id: path.parse(localImagePath).name,
      storage_bucket: bucket,
      storage_path: storagePath,
      local_image_path: localImagePath,
      source_dataset: row.source_dataset,
      original_label: row.original_label,
      status: "pending",
    });
  }

  for (let index = 0; index < rowsToInsert.length; index += 100) {
    const batch = rowsToInsert.slice(index, index + 100);
    const { error } = await supabase.from("labeling_images").insert(batch);
    if (error) throw error;
  }

  console.log(`dataset=${datasetSlug}`);
  console.log(`uploaded_files=${uploaded}`);
  console.log(`already_display_files=${alreadyDisplay}`);
  console.log(`display_max_dimension=${maxDisplayDimension}`);
  console.log(`display_jpeg_quality=${jpegQuality}`);
  console.log(`inserted_images=${rowsToInsert.length}`);
  console.log(`existing_images_preserved=${skippedExisting}`);
  console.log(`original_uploads_removed=${originalUploadsRemoved}`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
