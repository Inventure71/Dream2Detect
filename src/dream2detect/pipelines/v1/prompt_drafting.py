"""v1 prompt drafting compatibility wrappers.

The original implementation remains in :mod:`dream2detect.services` so existing
scripts and tests keep working. New pipeline work should import from the
versioned namespace.
"""

from dream2detect.services.prompt_drafting import (  # noqa: F401
    DraftedPrompt,
    PromptCandidate,
    PromptDraftResponse,
    draft_prompts_for_band,
    save_drafted_prompts,
)

