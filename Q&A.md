## Q1

Question: Is the real goal of Dream2Detect to prove a general synthetic-data concept, with package damage only as the case study, or is the goal mainly to build the best package-damage severity model possible?

Answer: The goal is to see if synthetic data is reliable enough to be used in a real-world scenario, especially for specific visual tasks that involve estimating severity such as "how much is it broken." Package damage is a concrete case study for that broader concept.

## Q2

Question: What should count as “reliable enough” for this class project: matching real-data-trained performance, beating simple baselines, or producing a defensible partial result like “works for coarse severity but not precise scores”?

Answer: Unresolved. Public datasets exist, especially Kaputt, but access/labels/scope need to be checked. The next decision is whether the project should train only on cardboard packages or broaden to many object types.

## Q3

Question: Should the core real-world comparison be based on a self-collected real test set, with public datasets used only as optional external validation?

Answer: Unresolved. The project should focus specifically on cardboard packages, not arbitrary objects. A possible alternative is comparing AI-generated image data against the Parcel3D synthetic/3D-generated dataset, making the project about whether 3D simulated data is better than AI-generated image data. Otherwise, the project needs a dataset specifically containing cardboard boxes and damage severity.

## Q4

Question: Do you want the central research question to stay about AI-generated synthetic data reliability, with Parcel3D as an optional comparison, or should the central question become AI-generated synthetic data vs 3D-simulated synthetic data?

Answer: Keep the central question about AI-generated synthetic data reliability for real cardboard-package severity estimation. Parcel3D can be optional or related work, but should not become the backbone because that would make the project too large.

## Q5

Question: Are you willing to collect and human-label a real cardboard-package test set yourselves, around 100-300 images?

Answer: Unresolved. The user asked whether there is a better or more fun challenge to train the model on instead of committing to collecting and labeling cardboard package damage.

## Q6

Question: Do you want the proof-of-concept to stay in the damage/severity world, or are you open to switching to a cleaner visual quantity estimation task like cup fill level?

Answer: Unresolved. The fill-level idea was not compelling enough; the user asked for better challenge candidates.

## Q7

Question: Do you want the challenge to feel more fun/memorable or more serious/industrial?

Answer: Unresolved. The choice should depend on the assignment instructions rather than only on what sounds fun.

## Q8

Question: Should we switch the case study from cardboard package damage to can crush/deformation severity?

Answer: Unresolved. The user wants to be convinced. The preferred task should be cool and should justify synthetic data because normal real-world labeling would be difficult, expensive, destructive, rare, or require too much effort.

## Q9

Question: Are you willing to switch from cardboard packages to phone screen crack severity as the case study?

Answer: Unresolved. The user's deeper goal is to prove that models can be trained from synthetic data from nothing. They proposed a possible task: estimating how clean a plate is using synthetic images of plates with different patterns, food residue, and stains.

## Q10

Question: Is the target variable “cleanliness” or “contamination/dirtiness”?

Answer: Unresolved. Plate contamination severity currently seems promising, but the user is concerned that dirty/clean plate datasets already exist online and the task may be too solved.

## Q11

Question: Should the plate project’s main challenge be pattern-confound robustness?

Answer: Unresolved. The user reframed the deeper goal as a "simulate and learn" challenge: testing whether synthetic generation can create many task-specific training examples on demand, supporting a future where autonomous robots learn new visual tasks when needed.

## Q12

Question: Should the docs be rewritten around the simulate-and-learn / on-demand robot perception framing, with plate contamination as the proof-of-concept task?

Answer: Yes. The project should be framed as simulate-and-learn / on-demand robot perception. The proof-of-concept task is plate contamination assessment for a robot deciding whether a plate is clean enough to stack, dirty enough for hand cleaning, or too dirty and should go to the washing machine.

## Q13

Question: Should the model output be primarily the robot action class, with the 0-100 contamination score as a secondary experiment?

Answer: Unresolved. The user asked first whether this project idea is feasible for the ML assignment and whether it has potential for a very high mark.

## Q14

Question: Should we officially replace the old package-damage docs with this new direction: simulate-and-learn plate contamination for robot action decisions?

Answer: Yes. Replace the package-damage framing with the robot plate-contamination framing, using action classification as the primary task and contamination severity as the secondary task.

## Q15

Question: What should be the project’s minimum success claim?

Answer: A model trained only on synthetic plate images should perform meaningfully better than trivial/classical baselines on real plate photos for the stack / hand-clean / dishwasher decision. Fine-grained contamination scoring is secondary and may fail, but that failure is still useful evidence.

## Q16

Question: Should the three robot action classes be defined by human action judgment or by a fixed 0-100 contamination threshold?

Answer: Define a detailed 0-100 contamination standard first, using 10% blocks. Each block should precisely describe what the plate should look like and how it should be evaluated. Then map percentage ranges into the three robot action labels: clean enough to put away, dirty enough to clean by hand, and too dirty so it must go to the washing machine. The model can still learn cleanliness/severity, while the three action labels simplify classification by converting score ranges into decisions.

## Q17

Question: Should contamination label descriptions and robot action labels live in the same standards file?

Answer: No. Split them. The contamination label descriptions should be a precise 0-100 visual dirtiness scale, written so a human or LLM can generate prompts for each range. Robot action labels should live in a separate file that maps contamination percentages to actions such as stack, hand clean, or dishwasher.

## Q18

Question: What are the main system workstreams for making the project work?

Answer: Split the system into two major workstreams: (1) data generation pipeline plus dataset handling, cleaning, standardization, entry validation, balancing, and quality checks; (2) model training, including model selection, training strategy, evaluation, and comparison.

## Q19

Question: What exact coarse severity label scheme should the package project use?

Answer: Use 4 coarse classes: `intact`, `minor`, `moderate`, `severe`.

## Q20

Question: What should the fine-grained score actually represent?

Answer: The fine-grained score should represent overall visible defect severity, not literal percentage broken. It should combine structural deformation, tears/holes, surface damage or dirt when it clearly affects condition, visible area affected, and overall practical seriousness. `0` means visually intact and `100` means maximally severe visible package defect.

## Q21

Question: How should the `0-100` fine score map into the 4 coarse classes?

Answer: Use `0-10 -> intact`, `11-35 -> minor`, `36-65 -> moderate`, and `66-100 -> severe`.

## Q22

Question: How should we split the Kaggle real dataset across `small real-only`, `fine-tuning`, and the final real test set?

Answer: Use a strict held-out real test first, then the remaining images as the small real-data branch. Default split: `60%` held-out real test, `20%` small real-only training, `20%` validation / fine-tuning support. Similar or near-duplicate images should stay in the same split.

## Q23

Question: Should we treat the Kaggle dataset as the only real domain in the official project, or should we plan to add your own box photos as a second real test source if time allows?

Answer: Use the Kaggle dataset as the official real evaluation source during development. Your own photos may be used later as an additional real-world stress test, but not during development or tuning.

## Q24

Question: For relabeling the Kaggle dataset, should every image get one human review after the ChatGPT suggestion, or should some images get a second human review?

Answer: Use one human review for every image, and require a second human review only for flagged cases such as uncertain ChatGPT suggestions, uncertain reviewer judgments, boundary cases, or images with multiple defect types that make severity ambiguous.

## Q25

Question: Should we include the same CNN trained only on the small real dataset as an official baseline?

Answer: Yes. Use the same CNN architecture across the official training-strategy comparison: real-only, synthetic-only, and synthetic plus small real fine-tuning.

## Q26

Question: Should the shared CNN be a small custom CNN from scratch or a standard pretrained backbone like ResNet/MobileNet?

Answer: Use a small custom CNN from scratch as the main official model. The goal of the project is to test learning from scratch, so pretrained backbones should not define the core experiment.

## Q27

Question: Should the main experiment use two separate CNNs or one multi-task CNN with two outputs?

Answer: Use two separate CNNs: one classifier for `intact / minor / moderate / severe`, and one regressor for the fine-grained severity score.

## Q28

Question: What exact classical baseline set should we commit to?

Answer: Use the following baseline lineup:
- trivial baseline: majority class for coarse labels, mean predictor for fine score
- classical classification baseline: HOG features + logistic regression
- classical regression baseline: HOG features + ridge regression

## Q29

Question: For the Kaggle real dataset, should the `20%` non-test portion be split into separate `real-only train` and `fine-tuning validation`, or should it be one small shared pool used for both purposes?

Answer: First merge the Kaggle real images back into one single candidate real dataset, relabel all images under the project's rubric, and only then split it into train/validation/test with balancing and leakage control. The split should be created after relabeling, not inherited from the original dataset structure.

## Q30

Question: After relabeling and rebuilding the Kaggle dataset, what exact split ratios should the final real-data setup use?

Answer: Use `60%` held-out real test, `20%` small real training, and `20%` real validation / fine-tuning support.

## Q31

Question: When we define “overall package defect severity,” should dirt/stain be part of that same target, together with dents, crushing, tears, and holes?

Answer: The project already defines the target as overall package defect severity. The missing work is not redefining the target, but defining in detail what each severity level means under that existing definition.

## Q32

Question: Do you want the detailed rubric to be built top-down from the 4 coarse classes first, and then map the fine score ranges into them?

Answer: Yes. Define the 4 coarse classes in detail first, then define the fine score bands so they map consistently into those classes.

## Q33

Question: What should `intact` mean exactly?

Answer: `intact` means no visible structural deformation, no tears, no holes, no crushed corners, no obvious dents, and only negligible surface marks that do not meaningfully affect package condition. Tiny printing wear, tiny dust, or trivial handling marks may be allowed. If a human hesitates and thinks a corner may already be damaged, it should not be labeled `intact`.

## Q34

Question: What should `minor` mean exactly?

Answer: `minor` means a small visible defect but no major structural compromise. This includes light denting or one small dent, one slightly softened or bent corner, a tiny tear or very short tear, or a superficial scratch, crease, or stain that clearly affects appearance. It excludes large holes, broad crushing, or major deformation. The package shape remains clearly normal overall. Damage is obvious, but the package still looks mostly intact.

## Q35

Question: What should `moderate` mean exactly?

Answer: `moderate` means clearly visible and meaningful damage. This includes noticeable denting, deformation, or multiple smaller defects; one clearly damaged corner or edge, or several smaller damaged areas; a visible tear that is more serious than a tiny surface tear; a small hole or opening; or partial crushing / clear loss of clean box geometry. Surface dirt or stain can contribute if it is substantial, but should not be the only reason unless it strongly affects condition. The package remains recognizable and not fully compromised, but its condition is clearly degraded.

## Q36

Question: What should `severe` mean exactly?

Answer: `severe` means major visible structural damage. This includes strong crushing, collapse, or major deformation; a large tear, major opening, or clear material failure; badly compromised corners or edges; and box geometry that is clearly broken rather than slightly softened. Heavy dirt or stain may contribute, but `severe` should usually require substantial structural or material damage, not just appearance. The package must remain recognizable as a package, but it should look clearly badly damaged.

## Q37

Question: Do you want the fine score to use only the 4 coarse class ranges, or do you want more detailed score bands inside each class?

Answer: Use smaller score bands inside the coarse classes. Default bands: `0-10` = `intact`; `11-20`, `21-30`, `31-35` inside `minor`; `36-45`, `46-55`, `56-65` inside `moderate`; `66-75`, `76-85`, `86-100` inside `severe`.

## Q38

Question: For synthetic data generation, should we prompt for exact scores like `47`, `73`, `88`, or should we prompt for score bands like `46-55 moderate`?

Answer: Prompt for score bands, not exact numeric scores. Each band should be defined very precisely, then used to ask an LLM to draft prompt sets for images in that band. Those prompts should be human-reviewed, tested on a small pilot batch, and only then used for larger batch image generation.
