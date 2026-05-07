# Dream2Detect

Dream2Detect is a comparative ML study of synthetic-to-real generalization for visible package/cardboard defect severity.

The core experiment compares:

- synthetic-only training
- small real-only training
- synthetic + small real fine-tuning

across:

- coarse severity classification
- fine severity regression

## Current Focus

The project is currently preparing labeled data.

The real-image source is Kaggle `redf0xwin/recognizing-defects-in-boxes-and-cardboard`. The original dataset labels and XML boxes are not the project ground truth. We manually relabel whole-image package severity into the Dream2Detect band system.

Primary label:

```text
score_band
```

Derived labels:

```text
coarse_class
representative_score
```

## Teammate Labeling

If you are helping label images, start here:

[docs/14-labeling-ui-teammate-guide.md](docs/14-labeling-ui-teammate-guide.md)

Short version:

```bash
cd apps/labeling-ui
npm install
cp .env.example .env
npm run dev
```

Then open:

```text
http://localhost:5174/
```

Use these browser-safe values in `apps/labeling-ui/.env`:

```bash
VITE_SUPABASE_URL=https://ourllofcpbmmihyjjugl.supabase.co
VITE_SUPABASE_ANON_KEY=sb_publishable_pQ_gWxgh0MFU-9qAaLfzDg_UHdu5aiC
VITE_DEFAULT_DATASET_SLUG=redf0xwin_boxes_cardboard
```

Do not edit CSV files directly while labeling. The UI saves labels to Supabase.

## Labeling Rule

Label the whole image by overall visible package condition.

Do not draw bounding boxes.

Use:

```text
score_band -> coarse_class -> representative_score
```

Examples:

```text
0-10   -> intact   -> 5
36-45  -> moderate -> 40
76-85  -> severe   -> 80
```

## Important Docs

- [Project understanding](docs/01-project-understanding.md)
- [Severity scale standards](docs/06-package-severity-scale-standards.md)
- [Real dataset relabeling](docs/07-real-dataset-relabeling.md)
- [Implementation plan](docs/08-implementation-plan.md)
- [Data contracts](docs/09-data-contracts.md)
- [Labeling UI teammate guide](docs/14-labeling-ui-teammate-guide.md)

## Repo Layout

```text
apps/labeling-ui/      Supabase-backed labeling web UI
data/                  local datasets, registries, templates, SQLite state
docs/                  project plan, labeling rubric, data contracts
generated_images/      generated synthetic image files
scripts/               project data/prompt/image helper scripts
src/dream2detect/      Python package code
```

## Admin Notes

Only the project owner should run admin sync commands.

Upload or refresh the online real-image queue:

```bash
cd apps/labeling-ui
npm run seed:real
```

Export reviewed labels back to the repo:

```bash
cd apps/labeling-ui
npm run export:real
```

The Supabase service-role key is required only for admin scripts. Never commit or share it.
