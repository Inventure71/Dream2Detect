# Prompt Writing Template

Use this template when asking an LLM to draft prompt candidates for one severity band.

## Instructions To The LLM

You are writing image-generation prompts for a machine-learning dataset about package/cardboard defect severity.

Follow these rules:

1. target only the requested severity band
2. keep the package clearly visible and central
3. describe visible package condition, not invisible hidden damage
4. avoid prompt text that asks for exact fake-precise numeric damage percentages
5. vary scene details without drifting away from the band definition
6. keep the object recognizable as a shipping package/cardboard box
7. do not add text overlays, labels, watermarks, or irrelevant dramatic clutter
8. respect the severity ceiling as strictly as the severity floor
9. do not use stronger damage phrases than the band allows
10. if the band forbids a large opening, exposed cavity, collapse, or severe crushing, the prompt must explicitly avoid those outcomes
11. preserve the structured assignment faithfully; do not silently move the damage to a different location or replace the assigned main defect with a different one
12. keep the assigned primary defect dominant, keep the assigned secondary defect subordinate, and do not invent a stronger third defect
13. if a field says a corner, edge, flap, face center, or side panel, the final prompt must keep the defect on that exact region
14. if the assignment says no visible label, do not invent a readable label; if it says barcode label only or small shipping label, keep that label secondary
15. keep the prompt realistic and varied, but do not use freedom in wording as an excuse to contradict the assignment
16. do not invent large readable printed package text, oversized warning stickers, dominant barcode blocks, or big background signage unless the assignment explicitly allows them, and even then keep them secondary and only partly readable
17. when the assignment allows a small label or barcode, keep it physically small, off to one side, and never make it the most visually salient element after the damage itself
18. keep the entire package visible in frame; do not use close-up crops, zoomed-in corners, macro shots, or partial-box framing
19. the assigned primary defect must remain visible from the assigned camera angle; do not choose a composition where the viewpoint hides the labeled damage

## Inputs

- band: `<score_band>`
- coarse class: `<coarse_class>`
- representative score: `<representative_score>`
- band spec file: `<band_spec_file>`
- primary defect profile: `<damage_profile_primary>`
- secondary defect profile: `<damage_profile_secondary>`
- primary damage location: `<damage_location_primary>`
- box form factor: `<box_form_factor>`
- box pattern: `<box_pattern>`
- label presence: `<label_presence>`
- tape profile: `<tape_profile>`
- background context: `<background_context>`
- camera angle: `<camera_angle>`
- lighting style: `<lighting_style>`

## Required Output

Return `N` prompt candidates in a structured list.

For each prompt include:

- short prompt title
- assignment key copied exactly from input
- final prompt text
- why it fits the target band
- what should be checked manually before generation
- what stronger cues must stay absent

## Hard Anchors vs Allowed Variation

Treat these as **hard anchors** that must stay semantically exact:

- primary defect profile
- secondary defect profile
- primary damage location
- box form factor
- box pattern
- label presence
- tape profile
- background context
- camera angle
- lighting style

Treat these as **allowed variation** that may change from prompt to prompt without breaking the assignment:

- precise cardboard shade or texture
- tiny cosmetic wear that stays below the assigned severity
- exact shadow shape
- minor camera distance while keeping the full box readable, central, and fully visible
- small scene details that do not introduce clutter or new defect meaning
- wording style, as long as the assignment meaning remains intact

If two prompts share a similar structured assignment, vary the non-label details rather than changing the label-defining defect.

## Reminder

The prompt must match the band visually.

It should not aim for a random damage level inside the class.

It should aim for the exact requested band.

Treat these phrases as escalation cues unless the band explicitly allows them:

- torn open
- ripped wide open
- exposed interior
- large hole
- collapsed corner
- major collapse
- severe crushing
- badly broken geometry

If the band is low or mid severity, prefer lower-force phrasing such as:

- frayed edge
- light tear
- partial tear
- softened corner
- dented face
- creased face
- noticeable but limited damage

## Assignment Fidelity Examples

Good:

- assignment says `front_face_center` and the prompt says the visible damage is centered on the main front face
- assignment says `bent_flap_or_edge` and the prompt keeps the main defect as a bent flap or damaged edge

Bad:

- assignment says `front_face_center` and the prompt turns the main defect into a damaged corner
- assignment says `bent_flap_or_edge` and the prompt replaces it with a dent as the main defect
- assignment says `no_visible_label` and the prompt invents a large readable shipping label
- assignment says `small_shipping_label` and the prompt invents a giant fragile sticker or a face-sized printed barcode block
- assignment says `packing_station` and the prompt invents a large readable wall sign or bold printed packaging slogan that dominates the frame
