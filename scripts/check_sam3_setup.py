from __future__ import annotations

import argparse
import inspect
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def parse_version(raw: str) -> tuple[int, ...]:
    parts: list[int] = []
    for chunk in raw.replace("+", ".").split("."):
        digits = "".join(char for char in chunk if char.isdigit())
        if digits == "":
            break
        parts.append(int(digits))
    return tuple(parts)


def format_status(ok: bool, label: str, detail: str) -> str:
    prefix = "PASS" if ok else "FAIL"
    return f"{prefix}: {label}: {detail}"


def hf_token_configured() -> bool:
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"):
        return True
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    return token_path.exists() and token_path.read_text(encoding="utf-8").strip() != ""


def resolve_requested_device(torch_module: Any, requested_device: str) -> str:
    if requested_device != "auto":
        return requested_device
    if torch_module.cuda.is_available():
        return "cuda"
    if torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


def collect_environment_checks(requested_device: str) -> tuple[bool, list[str], str | None]:
    messages: list[str] = []
    overall_ok = True

    python_version = sys.version_info[:3]
    python_ok = python_version >= (3, 12, 0)
    overall_ok = overall_ok and python_ok
    messages.append(
        format_status(
            python_ok,
            "Python",
            f"{python_version[0]}.{python_version[1]}.{python_version[2]} "
            "(requires >=3.12 for official SAM3)",
        )
    )

    try:
        import torch
    except Exception as exc:
        messages.append(format_status(False, "PyTorch import", f"{type(exc).__name__}: {exc}"))
        return False, messages, None

    torch_version = getattr(torch, "__version__", "unknown")
    torch_ok = parse_version(torch_version) >= (2, 7, 0)
    overall_ok = overall_ok and torch_ok
    messages.append(
        format_status(
            torch_ok,
            "PyTorch",
            f"{torch_version} (official SAM3 requires >=2.7)",
        )
    )

    selected_device = resolve_requested_device(torch, requested_device)
    messages.append(format_status(True, "Requested device", f"{requested_device} -> {selected_device}"))

    cuda_available = bool(torch.cuda.is_available())
    cuda_version = getattr(torch.version, "cuda", None)
    cuda_version_ok = cuda_version is not None and parse_version(str(cuda_version)) >= (12, 6)
    cuda_ok = selected_device != "cuda" or (cuda_available and cuda_version_ok)
    overall_ok = overall_ok and cuda_ok
    messages.append(
        format_status(
            cuda_ok,
            "CUDA",
            f"available={cuda_available}, torch.version.cuda={cuda_version!r} "
            "(required only for --device cuda)",
        )
    )

    mps_available = bool(torch.backends.mps.is_available())
    mps_built = bool(torch.backends.mps.is_built())
    mps_ok = selected_device != "mps" or (mps_built and mps_available)
    overall_ok = overall_ok and mps_ok
    messages.append(
        format_status(
            mps_ok,
            "MPS",
            f"built={mps_built}, available={mps_available} "
            "(experimental SAM3 path; not the official README runtime)",
        )
    )

    try:
        import sam3  # noqa: F401
    except Exception as exc:
        sam3_available = False
        sam3_detail = f"{type(exc).__name__}: {exc}"
    else:
        sam3_available = True
        sam3_detail = "importable"
    overall_ok = overall_ok and sam3_available
    messages.append(
        format_status(
            sam3_available,
            "sam3 package",
            sam3_detail,
        )
    )

    hf_ok = hf_token_configured()
    overall_ok = overall_ok and hf_ok
    messages.append(
        format_status(
            hf_ok,
            "Hugging Face token",
            "configured" if hf_ok else "not found in env or ~/.cache/huggingface/token",
        )
    )

    return overall_ok, messages, selected_device


def run_smoke(image_path: Path, prompt: str, device: str) -> None:
    from PIL import Image
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    if not image_path.exists():
        raise FileNotFoundError(f"Smoke image does not exist: {image_path}")

    model_kwargs = {}
    if device != "auto" and "device" in inspect.signature(build_sam3_image_model).parameters:
        model_kwargs["device"] = device
    model = build_sam3_image_model(**model_kwargs)

    processor_kwargs = {}
    if device != "auto" and "device" in inspect.signature(Sam3Processor).parameters:
        processor_kwargs["device"] = device
    processor = Sam3Processor(model, **processor_kwargs)
    image = Image.open(image_path).convert("RGB")
    state: Any = processor.set_image(image)
    output: dict[str, Any] = processor.set_text_prompt(state=state, prompt=prompt)
    boxes = output.get("boxes", [])
    scores = output.get("scores", [])
    print(f"SAM3 smoke prompt: {prompt!r}")
    print(f"SAM3 smoke boxes: {len(boxes)}")
    if len(boxes) > 0:
        first_score = scores[0].item() if hasattr(scores[0], "item") else scores[0]
        print(f"SAM3 first score: {float(first_score):.4f}")
        print(f"SAM3 first box: {boxes[0].tolist() if hasattr(boxes[0], 'tolist') else boxes[0]}")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check whether the current environment can run the official SAM3 image backend."
    )
    parser.add_argument(
        "--smoke-image",
        type=Path,
        default=None,
        help="Optional image path for an actual SAM3 text-prompt inference smoke.",
    )
    parser.add_argument("--prompt", default="cardboard box")
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help=(
            "Device to validate. auto selects cuda, then mps, then cpu. "
            "MPS is experimental for SAM3 and depends on a compatible SAM3 branch/build."
        ),
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    ok, messages, selected_device = collect_environment_checks(args.device)
    print("SAM3 environment check")
    for message in messages:
        print(message)

    if not ok:
        raise SystemExit(
            "SAM3 is not ready in this environment. For local Apple Silicon, use a "
            "SAM3 build that supports MPS and rerun with --device mps, or run the "
            "object-focused cache in explicit --crop-backend hybrid/center mode."
        )

    if args.smoke_image is not None:
        run_smoke(args.smoke_image.resolve(), args.prompt, selected_device or args.device)
    else:
        print("SAM3 environment checks passed. Add --smoke-image to run one inference.")


if __name__ == "__main__":
    main()
