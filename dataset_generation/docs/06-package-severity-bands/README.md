# Package Severity Band Specs

This folder contains the detailed band-by-band rubric for package/cardboard defect severity.

Each file is written to support two jobs:

1. **Prompt generation**
   Use the file to tell an LLM what a target band should look like before writing image prompts.
2. **Human labeling**
   Use the file to decide whether a real or synthetic image fits that band.

## Band List

- [00-10-intact.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/06-package-severity-bands/00-10-intact.md)
- [11-20-minor-low.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/06-package-severity-bands/11-20-minor-low.md)
- [21-30-minor-mid.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/06-package-severity-bands/21-30-minor-mid.md)
- [31-35-minor-high.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/06-package-severity-bands/31-35-minor-high.md)
- [36-45-moderate-low.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/06-package-severity-bands/36-45-moderate-low.md)
- [46-55-moderate-mid.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/06-package-severity-bands/46-55-moderate-mid.md)
- [56-65-moderate-high.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/06-package-severity-bands/56-65-moderate-high.md)
- [66-75-severe-low.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/06-package-severity-bands/66-75-severe-low.md)
- [76-85-severe-mid.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/06-package-severity-bands/76-85-severe-mid.md)
- [86-100-severe-high.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/06-package-severity-bands/86-100-severe-high.md)

## Shared Interpretation Rules

### 1. Judge overall visible condition

The target is **overall visible package defect severity**, not one isolated property.

That means the label should consider the combined visible effect of:

- deformation,
- dents,
- crushed corners,
- bent flaps,
- tears,
- holes,
- openings,
- stains or dirt when they meaningfully worsen condition.

### 2. Structural damage dominates cosmetic noise

If structural damage and cosmetic marks disagree, structural damage should dominate.

Examples:

- a package with a clear tear is not `intact` because it is otherwise clean,
- a package with strong crushing should not be downgraded because the rest of the faces are clean,
- a lightly dusty but geometrically normal box should not be pushed into a high band.

### 3. Dirt counts only when it materially worsens visible condition

Dirt, stains, scuffs, or residue can matter, but they should not dominate the score unless they are:

- substantial,
- obvious,
- visually degrading the package condition in a meaningful way.

Tiny dust, ordinary handling marks, or insignificant print wear should not drive the label.

### 4. Prompting should target bands, not fake-precise scores

Use:

- `score_band`
- `coarse_class`
- visual constraints from the matching file

Do not ask the image generator for exact numeric damage percentages.

### 4a. Prompting must respect severity ceilings

The most common failure mode is overshooting into a stronger band.

When writing prompts:

- do not use stronger damage language than the target band allows,
- do not imply exposed interior openings unless the band permits that,
- do not imply collapse unless the band permits that,
- do not let one dramatic defect quietly upgrade the whole image into a higher band.

Examples of escalation language that should be reserved for higher bands:

- torn open
- large opening
- exposed cavity
- collapsed corner
- major tear
- severe crushing
- broken geometry

Lower and mid bands should prefer language like:

- frayed corner
- limited tear
- noticeable dent
- bent edge
- softened corner
- partial crushing
- clearly damaged but not near-severe

### 5. Review before scale

The intended workflow is:

1. use a band file,
2. generate prompt drafts,
3. review the prompts,
4. create a small pilot image batch,
5. review the pilot against the same band file,
6. only then scale generation.

### 6. Borderline handling

If an image sits between two bands:

- choose the lower band if the stronger defect evidence is weak, uncertain, or barely visible,
- choose the higher band if the stronger defect signal is clear and materially changes perceived package condition.

### 7. Metadata expectation

Every accepted synthetic record should store:

- `score_band`
- `coarse_class`
- `band_spec_version`
- optional `representative_score`

The true target for generation is the band, not the exact score.
