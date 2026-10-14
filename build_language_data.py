import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    print("Install the required package with: pip install datasets")
    sys.exit(1)


# Hugging Face dataset
WILI_DATASET = "MartinThoma/wili_2018"

# Default languages for Jaimes
DEFAULT_LANGUAGES = [
    "en", "de", "fr", "es", "it",
    "pt", "nl", "pl", "ru", "ja", "zh"
]

SEED = 42

# Convert common ISO-639-3 codes to shorter codes.
# Other language codes are kept unchanged.
LANGUAGE_CODES = {
    "eng": "en",
    "deu": "de",
    "ger": "de",
    "fra": "fr",
    "fre": "fr",
    "spa": "es",
    "ita": "it",
    "por": "pt",
    "nld": "nl",
    "dut": "nl",
    "pol": "pl",
    "rus": "ru",
    "jpn": "ja",
    "zho": "zh",
    "chi": "zh",
    "cmn": "zh",
    "ara": "ar",
    "tur": "tr",
    "swe": "sv",
    "dan": "da",
    "nor": "no",
    "fin": "fi",
    "ces": "cs",
    "cze": "cs",
    "slk": "sk",
    "slo": "sk",
    "ukr": "uk",
    "ell": "el",
    "gre": "el",
    "heb": "he",
    "hin": "hi",
    "ind": "id",
    "kor": "ko",
    "hun": "hu",
    "ron": "ro",
    "rum": "ro",
    "bul": "bg",
    "hrv": "hr",
    "srp": "sr",
    "slv": "sl",
    "est": "et",
    "lit": "lt",
    "lav": "lv",
    "vie": "vi",
    "tha": "th",
    "cat": "ca",
    "eus": "eu",
    "baq": "eu",
    "glg": "gl",
    "gle": "ga",
    "isl": "is",
    "ice": "is",
    "fas": "fa",
    "per": "fa",
    "urd": "ur",
    "ben": "bn",
    "tam": "ta",
    "tel": "te",
    "mal": "ml",
    "mar": "mr",
    "guj": "gu",
    "pan": "pa",
    "swa": "sw",
    "afr": "af",
    "amh": "am",
    "sqi": "sq",
    "alb": "sq",
    "mkd": "mk",
    "mac": "mk",
}


def normalize_language(language):
    """Normalize common three-letter language codes."""
    language = str(language).strip().lower()
    return LANGUAGE_CODES.get(language, language)


def clean_text(text):
    """Remove excess whitespace and invalid empty text."""
    if not isinstance(text, str):
        return ""

    return " ".join(text.replace("\x00", " ").split()).strip()


def add_example(rows, seen, text, language, source, max_chars):
    """Validate an example and add it if it is not a duplicate."""
    text = clean_text(text)
    language = normalize_language(language)

    if not text or not language:
        return False

    # Ignore text without letters.
    if not any(char.isalpha() for char in text):
        return False

    # Match the current Jaimes model's input limit.
    if max_chars > 0:
        text = text[:max_chars].strip()

    if not text:
        return False

    # Deduplicate text within each language.
    key = (language, text.casefold())

    if key in seen:
        return False

    seen.add(key)

    rows.append({
        "text": text,
        "language": language,
        "source": source,
    })

    return True


def load_wili(rows, seen, languages, max_chars):
    """Download and load the WiLI-2018 training split."""
    print(f"\nLoading Hugging Face dataset: {WILI_DATASET}")

    dataset = load_dataset(WILI_DATASET, split="train")
    label_feature = dataset.features["label"]

    allowed = set(languages) if languages else None
    added = 0

    for item in dataset:
        raw_label = item["label"]

        # The label may be an integer ClassLabel or a string.
        if isinstance(raw_label, int):
            language = label_feature.int2str(raw_label)
        else:
            language = str(raw_label)

        language = normalize_language(language)

        if allowed is not None and language not in allowed:
            continue

        if add_example(
            rows,
            seen,
            item["sentence"],
            language,
            "WiLI-2018",
            max_chars,
        ):
            added += 1

    print(f"WiLI-2018: added {added:,} examples.")
    return added


def load_tatoeba(
    file_path,
    rows,
    seen,
    languages,
    max_chars,
    max_per_language,
):
    """
    Load Tatoeba's sentences export.

    Expected tab-separated columns:
        sentence_id, language_code, text
    """
    file_path = Path(file_path)

    if not file_path.exists():
        print(
            f"\nTatoeba file not found: {file_path}\n"
            "Skipping Tatoeba. Download the sentences export and "
            "provide its path using --tatoeba."
        )
        return 0

    print(f"\nReading Tatoeba: {file_path}")

    allowed = set(languages) if languages else None
    candidates = defaultdict(list)

    # Tatoeba's standard sentences export is tab-separated.
    with file_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.reader(file, delimiter="\t")

        for row in reader:
            if len(row) < 3:
                continue

            # Standard Tatoeba format:
            # sentence ID, language code, sentence text
            language = normalize_language(row[1])
            text = clean_text(row[2])

            if language in {"language", "lang"}:
                continue

            if allowed is not None and language not in allowed:
                continue

            if not text or not any(char.isalpha() for char in text):
                continue

            if max_chars > 0:
                text = text[:max_chars].strip()

            candidates[language].append(text)

    rng = random.Random(SEED)
    added = 0

    for language, texts in candidates.items():
        # Shuffle so the file's ordering doesn't determine which
        # examples are selected.
        rng.shuffle(texts)

        if max_per_language > 0:
            texts = texts[:max_per_language]

        language_added = 0

        for text in texts:
            if add_example(
                rows,
                seen,
                text,
                language,
                "Tatoeba",
                max_chars,
            ):
                added += 1
                language_added += 1

        print(f"  {language}: added {language_added:,} examples")

    print(f"Tatoeba: added {added:,} examples.")
    return added


def save_csv(rows, output_path):
    """Shuffle and save the combined dataset."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    random.Random(SEED).shuffle(rows)

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["text", "language", "source"],
            quoting=csv.QUOTE_MINIMAL,
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved CSV: {output_path.resolve()}")
    print(f"Total examples: {len(rows):,}")


def print_statistics(rows):
    """Print the number of examples per language and source."""
    language_counts = Counter(
        row["language"] for row in rows
    )

    source_counts = Counter(
        row["source"] for row in rows
    )

    print("\n--- Dataset statistics ---")

    print(f"Languages: {len(language_counts)}")

    print("\nExamples by source:")
    for source, count in sorted(source_counts.items()):
        print(f"  {source:15} {count:>10,}")

    print("\nExamples per language:")
    for language, count in sorted(language_counts.items()):
        print(f"  {language:5} {count:>10,}")


def main():
    parser = argparse.ArgumentParser(
        description="Combine WiLI-2018 and Tatoeba for Jaimes."
    )

    parser.add_argument(
        "--tatoeba",
        default="data/tatoeba/sentences.csv",
        help="Path to the tab-separated Tatoeba sentences export.",
    )

    parser.add_argument(
        "--output",
        default="data/languages.csv",
        help="Output CSV path.",
    )

    parser.add_argument(
        "--languages",
        default=",".join(DEFAULT_LANGUAGES),
        help=(
            "Comma-separated language codes. "
            "Use 'all' to include all languages from both sources."
        ),
    )

    parser.add_argument(
        "--max-tatoeba-per-language",
        type=int,
        default=3000,
        help="Maximum Tatoeba examples per language; 0 means unlimited.",
    )

    parser.add_argument(
        "--max-chars",
        type=int,
        default=160,
        help="Maximum text length; 0 means unlimited.",
    )

    args = parser.parse_args()

    if args.max_tatoeba_per_language < 0:
        parser.error("--max-tatoeba-per-language cannot be negative.")

    if args.max_chars < 0:
        parser.error("--max-chars cannot be negative.")

    if args.languages.strip().lower() == "all":
        languages = None
    else:
        languages = [
            normalize_language(code)
            for code in args.languages.split(",")
            if code.strip()
        ]

    rows = []
    seen = set()

    # First source: Hugging Face.
    load_wili(
        rows=rows,
        seen=seen,
        languages=languages,
        max_chars=args.max_chars,
    )

    # Second source: Tatoeba.
    load_tatoeba(
        file_path=args.tatoeba,
        rows=rows,
        seen=seen,
        languages=languages,
        max_chars=args.max_chars,
        max_per_language=args.max_tatoeba_per_language,
    )

    if not rows:
        print("\nNo examples were collected.")
        sys.exit(1)

    save_csv(rows, args.output)
    print_statistics(rows)


if __name__ == "__main__":
    main()