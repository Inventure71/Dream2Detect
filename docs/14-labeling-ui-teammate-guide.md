# Labeling UI Guide

This guide explains how to run the Dream2Detect labeling UI locally.

The shared source of truth is Supabase. Each teammate runs the web app on their own machine, but labels are saved to the same online database.

## Teammate Quick Start

Use this section if your job is only to label images.

### What You Need

- Node.js installed
- access to this repo
- an email address you can open during sign-in

You do **not** need a Supabase admin account or service-role key.

### One-Time Setup

From the repo root:

```bash
cd apps/labeling-ui
npm install
```

Create a local `.env` file:

```bash
cp .env.example .env
```

Edit `apps/labeling-ui/.env` so these values are set:

```bash
VITE_SUPABASE_URL=https://ourllofcpbmmihyjjugl.supabase.co
VITE_SUPABASE_ANON_KEY=sb_publishable_pQ_gWxgh0MFU-9qAaLfzDg_UHdu5aiC
VITE_DEFAULT_DATASET_SLUG=redf0xwin_boxes_cardboard
```

The `.env` file is intentionally ignored by git.

### Start The App

From `apps/labeling-ui`:

```bash
npm run dev
```

Open:

```text
http://localhost:5174/
```

### Sign In

1. Enter your email.
2. Click sign in.
3. Open the newest email from Supabase.
4. Click the magic link once.

Magic links are one-time use. If you see `otp_expired`, go back to:

```text
http://localhost:5174/
```

and request a fresh link.

### Label Images

1. Click `Get next image`.
2. Inspect the image.
3. Use the top band examples and the left-side band descriptions.
4. Select the best `score_band`.
5. Add notes only when useful.
6. Check `Boundary or uncertain` if the case is unclear.
7. Click:
   - `Save Next` for normal labels
   - `Second Review` for uncertain/borderline cases
   - `Reject` if the image is unusable
   - `Skip` if you do not want to label the current image

The system automatically derives:

- `coarse_class`
- `representative_score`

Do not edit CSV files manually while labeling.

### Multi-User Behavior

When you click `Get next image`, the system locks that image for you.

This prevents two teammates from labeling the same image at the same time.

If you skip an image or click `Next` without saving, the lock is released and another teammate can label it.

If you close the browser without saving, the lock expires after about 30 minutes.

### Labeling Rule

Label the whole image by overall visible package condition.

Do **not** draw bounding boxes.

Use the band-first system:

```text
score_band -> coarse_class -> representative_score
```

Examples:

```text
0-10   -> intact   -> 5
36-45  -> moderate -> 40
76-85  -> severe   -> 80
```

### What Not To Do

- Do not edit `data/registries/real_relabel_registry.csv` directly.
- Do not run `npm run seed:real`.
- Do not run `npm run export:real` unless you are syncing final labels back to the repo.
- Do not share or request the Supabase service-role key.

## Admin / Project Owner Notes

Only use this section if you are maintaining the shared dataset, uploading new images, or exporting labels back into the repo.

### Service Role Key

Admin scripts need:

```bash
SUPABASE_SERVICE_ROLE_KEY=...
```

This key must stay private. It belongs only in your local ignored `.env` file.

Never paste the service-role key into docs, screenshots, commits, Discord, or the frontend app.

Browser-safe values are allowed in the docs:

```bash
VITE_SUPABASE_URL=https://ourllofcpbmmihyjjugl.supabase.co
VITE_SUPABASE_ANON_KEY=sb_publishable_pQ_gWxgh0MFU-9qAaLfzDg_UHdu5aiC
VITE_DEFAULT_DATASET_SLUG=redf0xwin_boxes_cardboard
```

### Export Labels

Only one project owner should periodically export labels back to the repo:

```bash
cd apps/labeling-ui
npm run export:real
```

This updates:

```text
data/registries/real_relabel_registry.csv
```

from the shared Supabase database.

### Upload Or Refresh The Online Dataset

Only one project owner should upload or refresh online images:

```bash
cd apps/labeling-ui
npm run seed:real
```

This reads the local real registry, uploads compressed display images to Supabase Storage, and updates the online image queue.
