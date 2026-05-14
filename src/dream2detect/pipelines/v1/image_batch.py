"""v1 image batch compatibility wrappers."""

from dream2detect.services.batch_image_generation import (  # noqa: F401
    BATCH_ENDPOINT,
    PreparedImageBatch,
    build_image_batch_request,
    custom_id_for_prompt_id,
    ingest_completed_image_batch,
    prepare_image_generation_batch,
    prompt_id_from_custom_id,
    refresh_image_batch_status,
    submit_prepared_image_batch,
    utc_timestamp,
)

