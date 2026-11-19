"""OCR on macOS through the Vision framework.

Windows runs ``Windows.Media.Ocr`` via PowerShell. macOS has the Vision
framework built in, so text recognition runs through ``VNRecognizeTextRequest``
(pyobjc) with no external binary. When pyobjc is unavailable the module falls
back to the ``tesseract`` command line tool.

The public contract matches the Windows batch runner: a list of PIL images in,
a list of per-image lists of text lines out.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

# Vision recognition level. ACCURATE is what the Windows OCR engine
# corresponds to and matters for URL-shaped strings.
_RECOGNITION_LEVEL_ACCURATE = 0
_RECOGNITION_LEVEL_FAST = 1

# Languages the bridge cares about. Empty means "let Vision use its default",
# which follows the user's macOS language settings.
_DEFAULT_LANGUAGES = ("en-US",)


def _require_vision():
    """Return the pyobjc Vision and CoreGraphics modules, or raise."""
    import Quartz
    import Vision

    return Vision, Quartz


def _supported() -> bool:
    try:
        _require_vision()
    except Exception:
        return False
    return True


def _make_handler(cg_image, vision):
    """Build the pyobjc Vision request handler for one CGImage."""
    return vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg_image, None)


def _cg_image_from_pil(pil_image, quartz):
    """Convert a PIL image into a CGImage.

    The bridge hands us grayscale ("L") binarized crops as often as RGB, and
    the preprocessed crops are heavily upscaled (2.5x-3.6x), so this stays on
    the normal path rather than trying to be clever.
    """
    rgb = pil_image if pil_image.mode == "RGB" else pil_image.convert("RGB")
    width, height = rgb.size

    # NSBitmapImageRep wants the raw bytes in a known layout; PIL gives us
    # top-down RGB which is what CGImage expects for the common case.
    data = rgb.tobytes()
    provider = quartz.CGDataProviderCreateWithData(None, data, len(data), None)
    if provider is None:
        raise RuntimeError("Could not create a CGDataProvider from the image data.")

    color_space = quartz.CGColorSpaceCreateDeviceRGB()
    return quartz.CGImageCreate(
        width,
        height,
        8,
        24,
        width * 3,
        color_space,
        quartz.kCGImageAlphaNone,
        provider,
        None,
        False,
        quartz.kCGRenderingIntentDefault,
    )


def _recognize_text(pil_image) -> list[str]:
    """Recognize text in one PIL image, returning the text lines top to bottom."""
    try:
        vision, quartz = _require_vision()
    except Exception as exc:
        raise RuntimeError(f"pyobjc Vision framework unavailable: {exc}") from exc

    cg_image = _cg_image_from_pil(pil_image, quartz)
    if cg_image is None:
        return []

    request = vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(_RECOGNITION_LEVEL_ACCURATE)
    request.setUsesLanguageCorrection_(True)

    languages = []
    for lang in _DEFAULT_LANGUAGES:
        languages.append(lang)
    if languages:
        try:
            request.setRecognitionLanguages_(languages)
        except Exception:
            pass

    handler = _make_handler(cg_image, vision)
    ok, error = handler.performRequests_error_([request], None)
    if not ok:
        raise RuntimeError(f"Vision request failed: {error}")

    results = request.results()
    if not results:
        return []

    lines: list[tuple[float, str]] = []
    for observation in results:
        candidates = observation.topCandidates_(1)
        if not candidates:
            continue
        candidate = candidates[0]
        text = str(candidate.string())
        if not text.strip():
            continue
        # Vision reports the bounding box in normalised, bottom-left origin
        # coordinates; the negative Y is the distance from the bottom edge.
        box = observation.boundingBox()
        top = -float(box.origin.y) if box is not None else 0.0
        lines.append((top, text))

    # Reading order: top to bottom, then left to right.
    lines.sort(key=lambda item: item[0])
    return [text for _top, text in lines]


def recognize_text_batch(images) -> list[list[str]]:
    """Recognize text in a batch of PIL images.

    Returns one list of text lines per input image, in the same order.
    An image that yields nothing contributes an empty list -- the bridge
    treats "no text" and "OCR failed" the same way.
    """
    images = list(images or [])
    if not images:
        return []

    if not _supported():
        logger.info("pyobjc Vision framework unavailable, falling back to tesseract")
        return recognize_text_batch_tesseract(images)

    results: list[list[str]] = []
    for image in images:
        try:
            results.append(_recognize_text(image))
        except Exception as exc:
            logger.warning("Vision OCR failed for one image: %s", exc)
            results.append([])
    return results


def recognize_text(image) -> list[str]:
    """Recognize text in a single PIL image."""
    batch = recognize_text_batch([image])
    return batch[0] if batch else []


# ---------------------------------------------------------------------------
# tesseract fallback
# ---------------------------------------------------------------------------


def _tesseract_lines(image, path: Path) -> list[str]:
    """Run tesseract on one image file and return its text lines."""
    exe = shutil.which("tesseract")
    if not exe:
        return []
    try:
        result = subprocess.run(
            [exe, str(path), "stdout", "-l", "eng"],
            capture_output=True,
            timeout=15.0,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("tesseract failed: %s", exc)
        return []
    if result.returncode != 0:
        logger.warning(
            "tesseract exited %s: %s",
            result.returncode,
            result.stderr.decode("utf-8", "ignore").strip(),
        )
        return []
    text = result.stdout.decode("utf-8", "replace")
    return [line.strip() for line in text.splitlines() if line.strip()]


def recognize_text_batch_tesseract(images) -> list[list[str]]:
    """Recognize text with tesseract, used when pyobjc is unavailable."""
    images = list(images or [])
    if not images:
        return []

    results: list[list[str]] = []
    with tempfile.TemporaryDirectory(prefix="iphone_link_ocr_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        for index, image in enumerate(images):
            out_path = tmp_path / f"ocr_{index}.png"
            try:
                image.save(out_path, format="PNG")
            except Exception as exc:
                logger.warning("tesseract fallback could not save image: %s", exc)
                results.append([])
                continue
            results.append(_tesseract_lines(image, out_path))
    return results
