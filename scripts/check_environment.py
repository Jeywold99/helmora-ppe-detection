"""
Quick environment check before downloading data or training.
Run this first: python scripts/check_environment.py
"""

import importlib
import sys


def check(label, fn):
    try:
        result = fn()
        print(f"[OK]   {label}: {result}")
        return True
    except Exception as e:
        print(f"[FAIL] {label}: {e}")
        return False


def main():
    print(f"Python: {sys.version.split()[0]}\n")

    ok = True
    ok &= check("torch import", lambda: __import__("torch").__version__)

    import torch
    ok &= check("CUDA available", lambda: torch.cuda.is_available())
    if torch.cuda.is_available():
        check("GPU name", lambda: torch.cuda.get_device_name(0))
        check("GPU memory (GB)", lambda: round(
            torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
    else:
        print("[warn] No CUDA GPU detected -- training will be slow (CPU) or "
              "use MPS on Apple Silicon. Fine for this smoke test, not for a "
              "real training run.")

    ok &= check("transformers import", lambda: __import__("transformers").__version__)
    ok &= check("pycocotools import", lambda: importlib.import_module("pycocotools").__name__)
    ok &= check("opencv import", lambda: __import__("cv2").__version__)

    print("\nAttempting to download a small pretrained RT-DETR checkpoint "
          "from Hugging Face (tests your internet connection + model access)...")
    ok &= check(
        "RT-DETR checkpoint download",
        lambda: __import__("transformers").AutoModelForObjectDetection.from_pretrained(
            "PekingU/rtdetr_r18vd"
        ).config.model_type,
    )

    print("\n" + ("All checks passed -- ready to proceed." if ok else
                  "Some checks failed -- fix these before downloading data or training."))


if __name__ == "__main__":
    main()
