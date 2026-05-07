import { createClient } from "@supabase/supabase-js";
import { writeFile } from "node:fs/promises";
import path from "node:path";
import "./load-env.mjs";
import { stringifyCsv } from "./csv-utils.mjs";

const appRoot = process.cwd();
const repoRoot = process.env.DREAM2DETECT_REPO_ROOT ?? path.resolve(appRoot, "../..");
const outputPath =
  process.env.REAL_REGISTRY_PATH ??
  path.join(repoRoot, "data/registries/real_relabel_registry.csv");

const supabaseUrl = process.env.SUPABASE_URL;
const serviceRoleKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

if (!supabaseUrl || !serviceRoleKey) {
  throw new Error("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required.");
}

const supabase = createClient(supabaseUrl, serviceRoleKey, {
  auth: { persistSession: false },
});

const headers = [
  "image_id",
  "image_path",
  "source_dataset",
  "original_label",
  "llm_score_band",
  "llm_representative_score",
  "llm_coarse_class",
  "llm_reasoning",
  "human_score_band",
  "human_representative_score",
  "human_coarse_class",
  "final_score_band",
  "final_representative_score",
  "final_coarse_class",
  "review_status",
  "disagreement_flag",
  "review_notes",
  "split",
];

async function main() {
  const { data, error } = await supabase
    .from("real_relabel_export")
    .select("*")
    .order("image_id", { ascending: true });

  if (error) throw error;

  await writeFile(outputPath, stringifyCsv(data, headers), "utf8");
  console.log(`exported_rows=${data.length}`);
  console.log(`output=${outputPath}`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
