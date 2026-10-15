"""Evaluate OCR on a labelled, licence-documented CSV manifest."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from collections.abc import Sequence

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from backend.services.ocr_service import recognize_tamil_text  # noqa: E402
from backend.services.preprocessing import preprocess_image  # noqa: E402


def edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, reference_character in enumerate(reference, start=1):
        current = [row]
        for column, hypothesis_character in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[column - 1] + 1,
                    previous[column] + 1,
                    previous[column - 1]
                    + (reference_character != hypothesis_character),
                )
            )
        previous = current
    return previous[-1]


def normalized_text(text: str) -> str:
    return " ".join(text.split())


def error_rates(reference: str, hypothesis: str) -> tuple[float, float]:
    reference = normalized_text(reference)
    hypothesis = normalized_text(hypothesis)
    cer = edit_distance(reference, hypothesis) / max(1, len(reference))
    reference_words = reference.split()
    hypothesis_words = hypothesis.split()
    wer = edit_distance(reference_words, hypothesis_words)
    wer /= max(1, len(reference_words))
    return cer, wer


def manifest_rows(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    required = {"image_path", "tamil_text", "category", "split", "license"}
    if not rows:
        raise ValueError("The manifest has no rows.")
    if not required.issubset(rows[0]):
        raise ValueError(
            "Manifest columns must include: " + ", ".join(sorted(required))
        )
    if any(not row.get("license", "").strip() for row in rows):
        raise ValueError("Every image must have a recorded permission or licence.")
    return rows


def ensure_no_split_leakage(rows: list[dict[str, str]], root: Path) -> None:
    hashes_by_split: dict[str, set[str]] = {}
    for row in rows:
        image_path = (root / row["image_path"]).resolve()
        try:
            image_path.relative_to(root.resolve())
        except ValueError as exc:
            raise ValueError("Manifest image paths must stay within the dataset folder.") from exc
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        split = row["split"].strip().lower()
        previous_splits = [
            name for name, hashes in hashes_by_split.items() if digest in hashes
        ]
        if previous_splits and split not in previous_splits:
            raise ValueError(
                f"Duplicate image leakage detected across {previous_splits[0]} and {split}."
            )
        hashes_by_split.setdefault(split, set()).add(digest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Dataset folder containing manifest.csv")
    parser.add_argument("--split", default="test", help="Split to score (default: test)")
    parser.add_argument("--language", choices=("tam", "tam+eng"), default="tam")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    manifest = args.dataset / "manifest.csv"
    try:
        rows = manifest_rows(manifest)
        ensure_no_split_leakage(rows, args.dataset)
        selected = [
            row for row in rows if row["split"].strip().lower() == args.split.lower()
        ]
        if not selected:
            raise ValueError(f"No rows found for split {args.split!r}.")

        per_image = []
        for row in selected:
            image_path = (args.dataset / row["image_path"]).resolve()
            prepared = preprocess_image(image_path.read_bytes())
            original_result = recognize_tamil_text(
                prepared.original,
                language=args.language,
            )
            processed_result = recognize_tamil_text(
                prepared.processed,
                language=args.language,
            )
            original_cer, original_wer = error_rates(row["tamil_text"], original_result.text)
            processed_cer, processed_wer = error_rates(row["tamil_text"], processed_result.text)
            per_image.append(
                {
                    "image_path": row["image_path"],
                    "category": row["category"],
                    "reference_text": row["tamil_text"],
                    "original_text": original_result.text,
                    "processed_text": processed_result.text,
                    "original_cer": original_cer,
                    "processed_cer": processed_cer,
                    "original_wer": original_wer,
                    "processed_wer": processed_wer,
                }
            )
    except (OSError, ValueError, KeyError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1

    summary = {
        "split": args.split,
        "image_count": len(per_image),
        "mean_original_cer": sum(row["original_cer"] for row in per_image) / len(per_image),
        "mean_processed_cer": sum(row["processed_cer"] for row in per_image) / len(per_image),
        "mean_original_wer": sum(row["original_wer"] for row in per_image) / len(per_image),
        "mean_processed_wer": sum(row["processed_wer"] for row in per_image) / len(per_image),
        "items": per_image,
    }
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
