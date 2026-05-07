# Dream2Detect Labeling UI

Supabase-backed web UI for collaborative relabeling and synthetic-image QC.

The active online source of truth is Supabase. The repo CSV files are seeded into Supabase before labeling and exported back after labeling.

## Structure

```text
apps/labeling-ui/
  supabase/schema.sql              # shared database schema, lock functions, export view
  scripts/seed-real-registry.mjs   # uploads local real images and seeds image rows
  scripts/export-real-labels.mjs   # exports final online labels back to repo CSV
  src/                             # Vite React app
```

## Setup

For teammate labeling instructions, use:

- `../../docs/14-labeling-ui-teammate-guide.md`

Admin setup:

1. Create a Supabase project.
2. In the Supabase SQL editor, run `supabase/schema.sql`.
3. Copy `.env.example` to `.env`.
4. Fill in:

```text
VITE_SUPABASE_URL
VITE_SUPABASE_ANON_KEY
SUPABASE_URL
SUPABASE_SERVICE_ROLE_KEY
```

The browser app only uses the anon/publishable key. The service-role key is only for admin local scripts.

Current Supabase project:

```text
Project: dream2detect-labeling
URL: https://ourllofcpbmmihyjjugl.supabase.co
Bucket: dream2detect-images
```

## Upload The Current Real Dataset

Admin only. Teammates should not run this.

From this folder:

```bash
npm install
npm run seed:real
```

The seed script:

- reads `../../data/registries/real_relabel_registry.csv`
- creates the storage bucket if needed
- uploads compressed display images to Supabase Storage
- inserts missing `labeling_images` rows
- preserves existing online image rows and review state

By default the uploaded display copy is capped at `1280px` on the longest side
with JPEG quality `72`. Local source images are not modified. To change this:

```text
LABELING_MAX_IMAGE_DIMENSION=1280
LABELING_JPEG_QUALITY=72
```

## Run The UI

```bash
npm run dev
```

Open the Vite URL, sign in by email, then label images from the configured dataset slug.

Magic-link sign-in links are one-time use. If the browser shows
`otp_expired`, request a fresh email link and open only the newest link.

## Export Labels Back To The Repo

Admin only. Teammates should not run this.

```bash
npm run export:real
```

This rewrites:

```text
../../data/registries/real_relabel_registry.csv
```

from the Supabase `real_relabel_export` view.

## Add Another Dataset

1. Put the new images in a dataset-specific folder under `data/real/source/`.
2. Create or update a registry CSV with `image_id`, `image_path`, `source_dataset`, and any source hints in `original_label`.
3. Set a new dataset slug before seeding:

```bash
SUPABASE_DATASET_SLUG=my_new_dataset \
SUPABASE_DATASET_NAME="My New Dataset" \
REAL_REGISTRY_PATH=/absolute/path/to/my_registry.csv \
npm run seed:real
```

The UI can switch datasets by changing the dataset slug field in the top bar.

## Concurrency Model

The UI uses Supabase RPC functions:

- `claim_next_image`
- `save_label_review`
- `release_image_lock`

Each claimed image gets a 30-minute lock. Saves include the image `version`, so stale saves are rejected instead of overwriting newer work.

Reviews are stored in `label_reviews`; the current resolved label is stored in `final_labels`.

Clicking `Next` without saving first releases the current image lock before
claiming another image. This keeps skipped images available for other reviewers
instead of leaving abandoned locks.
