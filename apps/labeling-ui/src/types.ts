export type LabelingImage = {
  id: string;
  dataset_id: string;
  image_id: string;
  source_image_id: string;
  storage_bucket: string;
  storage_path: string;
  local_image_path: string | null;
  source_dataset: string | null;
  original_label: string | null;
  prompt_id: string | null;
  prompt_text: string | null;
  intended_score_band: string | null;
  intended_coarse_class: string | null;
  intended_representative_score: number | null;
  status: "pending" | "in_review" | "labeled" | "needs_second_review" | "rejected";
  lock_owner: string | null;
  lock_reviewer_name: string | null;
  lock_expires_at: string | null;
  version: number;
};

export type DatasetProgress = {
  slug: string;
  name: string;
  kind: "real" | "synthetic";
  total: number;
  pending: number;
  in_review: number;
  labeled: number;
  needs_second_review: number;
  rejected: number;
};

export type SaveDecision = "approved" | "corrected" | "needs_second_review" | "rejected";
