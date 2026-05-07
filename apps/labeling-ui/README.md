# Labeling UI Admin Notes

Teammate instructions:

[../../INSTRUCTIONS_FOR_LABELLING.md](../../INSTRUCTIONS_FOR_LABELLING.md)

## Current Supabase Project

```text
Project: dream2detect-labeling
URL: https://ourllofcpbmmihyjjugl.supabase.co
Bucket: dream2detect-images
Dataset slug: redf0xwin_boxes_cardboard
```

## Files

```text
supabase/schema.sql
scripts/seed-real-registry.mjs
scripts/export-real-labels.mjs
scripts/generate-band-examples.mjs
src/
```

## Admin Environment

Create `apps/labeling-ui/.env`.

Public frontend values:

```bash
VITE_SUPABASE_URL=https://ourllofcpbmmihyjjugl.supabase.co
VITE_SUPABASE_ANON_KEY=sb_publishable_pQ_gWxgh0MFU-9qAaLfzDg_UHdu5aiC
VITE_DEFAULT_DATASET_SLUG=redf0xwin_boxes_cardboard
```

Admin-only value:

```bash
SUPABASE_SERVICE_ROLE_KEY=...
```

Never commit or share the service-role key.

## Apply Schema

Run `supabase/schema.sql` as a Supabase migration.

## Upload / Refresh Real Images

```bash
npm install
npm run seed:real
```

This reads:

```text
../../data/registries/real_relabel_registry.csv
```

and uploads compressed display images to Supabase Storage.

Default display settings:

```bash
LABELING_MAX_IMAGE_DIMENSION=1280
LABELING_JPEG_QUALITY=72
```

## Export Labels Back To Repo

```bash
npm run export:real
```

This rewrites:

```text
../../data/registries/real_relabel_registry.csv
```

from Supabase.

## Generate Band Example Thumbnails

```bash
npm run generate:band-examples
```

Outputs:

```text
public/band-examples/
```
