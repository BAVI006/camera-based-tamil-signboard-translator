# Tamil signboard evaluation data

No photographs or evaluation results are included in this repository.

## Collection and annotation

1. Collect photos only when the photographer or rights holder permits your
   intended use. Record the exact licence/permission and attribution for every
   image. Do not assume a publicly visible photo is reusable.
2. Avoid capturing identifiable people, vehicle plates, private addresses, or
   other unnecessary personal data. Remove images that cannot be safely shared.
3. Ask a fluent Tamil reader to record the exact visible Tamil text, preserving
   line order. Have a second reviewer verify each transcription.
4. Have reviewers assign one category from: `transport`, `directions_places`,
   `healthcare`, `education`, `shops_businesses`, `warnings_prohibitions`,
   `government_public_services`, `other_unknown`.
5. Split by unique image/signboard identity into `train`, `validation`, and
   `test` before tuning anything. Keep photographs of the same sign and near
   duplicates in only one split.
6. Store images within this dataset directory and create `manifest.csv` with:
   `image_path,tamil_text,category,split,license`.
7. Do not publish collected photos or derived records unless the permission
   explicitly allows it.

## OCR evaluation

From the `SignSenseAI` folder, score a held-out test split:

```cmd
..\.venv\Scripts\python.exe backend\scripts\evaluate_ocr.py backend\data\dataset --split test --language tam+eng --output backend\data\dataset\test-results.json
```

The script rejects missing licence metadata, paths escaping the dataset folder,
and exact duplicate images placed across different splits. It reports
Character Error Rate (CER) and Word Error Rate (WER) for both the original
orientation-corrected image and the preprocessed image. Lower error rates are
better. Do not use the test split for tuning.

For context classification, create a separately reviewed category label for
each test item and report per-category precision, recall, F1, and macro-F1.
Translation quality needs human bilingual review for meaning preservation,
critical safety content, and named entities; do not use model confidence as a
substitute for translation evaluation.

Fine-tuning is not enabled. First collect a rights-cleared dataset, freeze a
baseline, inspect error types, then consider training only if results show a
repeatable gap that preprocessing or inference configuration cannot address.
