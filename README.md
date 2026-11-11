# Aligning Lecture Generation with Pedagogical Structure

Code and instructions for a curated dataset of 100 complete lectures for *Aligning Lecture Generation with Pedagogical Structure*.

Pedagogy-aware fine-tuning (PAFT) learns pedagogical tags and slide-grounded narration jointly. The code supports PAFT, conventional SFT, guided and unguided zero-shot, few-shot, and chain-of-thought generation with Qwen3.5 vision-language models.

## Contents

- `src/paft.py`: pedagogy-aware fine-tuning with joint tag and narration supervision.
- `src/train.py`: conventional SFT with plain-transcript supervision.
- `src/generate.py`: prompting baselines and fine-tuned generation.
- `src/annotate.py`: sentence-span annotation using Gemini.
- `src/prepare_data.py`: conversion from lecture CSV files to JSONL.
- `src/scrape_nptel.py`: NPTEL transcript-link collection and optional PDF downloading.
- `prompts/`: prompting-baseline and annotation instructions.
- `data/`: destination for the numbered dataset ZIP uploads.

Use the scripts under `src/` in the commands below. Copies of several scripts also exist at the repository root.

The following dataset paths are created after extracting all ZIP parts:

- `data/lectures/`: 100 lecture folders, each containing ordered slide images, plain narration, annotated narration, and metadata.
- `data/records/`: one JSONL file containing 100 complete lecture records.
- `data/lectures.csv`: lecture index with subject, slide count, word count, and transcript locations.

## Download and extract the dataset

The dataset is packaged as **27 independent ZIP archives**, named `data_part_01.zip` through `data_part_27.zip`. Each archive is below 25,000,000 bytes (approximately 17.9–22.9 MB), allowing upload through GitHub's web interface.

**Availability:** at the latest repository check, `data/` contained only `.gitkeep`; the ZIP parts were not yet committed to `main`. The instructions below apply once all 27 parts appear in [the data folder](data/). A placeholder file does not contain the dataset.

Download the repository using **Code → Download ZIP** and extract that outer ZIP, or clone it:

```bash
git clone https://github.com/sonalijolly804-debug/education.git
cd education
```

Run extraction from the repository root (the folder containing `src/`, `prompts/`, and `README.md`). Keep the 27 dataset ZIPs inside `data/`. GitHub does not unpack uploaded archives automatically.

**Linux/macOS (Bash):**

```bash
# Check that every expected part exists before extracting anything.
(
  for i in {1..27}; do
    printf -v part 'data/data_part_%02d.zip' "$i"
    [ -f "$part" ] || { echo "Missing: $part"; exit 1; }
  done
  for i in {1..27}; do
    printf -v part 'data/data_part_%02d.zip' "$i"
    unzip -o "$part" -d . || exit 1
  done
)
```

**Windows (PowerShell):**

```powershell
$parts = 1..27 | ForEach-Object { "data/data_part_{0:D2}.zip" -f $_ }
$missing = @($parts | Where-Object { -not (Test-Path -LiteralPath $_) })
if ($missing.Count -gt 0) { throw ("Missing ZIP parts: " + ($missing -join ", ")) }
foreach ($part in $parts) {
    Expand-Archive -LiteralPath $part -DestinationPath . -Force -ErrorAction Stop
}
```

Extract every part into the **same repository root**, not into `data/`: archive entries already start with `data/`. These are standalone ZIPs; do not concatenate them. A lecture may span multiple parts, so all parts are required for the complete dataset.

After extraction, expect `data/lectures.csv`, `data/selection.json`, `data/records/lectures_01.jsonl`, and 100 folders under `data/lectures/`. Together they contain **2,017 files**, including **1,714 PNG slides**, and occupy about **585 MiB** uncompressed. Original lecture IDs are retained and are not numbered consecutively from 1 to 100.

## Setup

Use Python 3.10 or later and a CUDA environment supported by Unsloth. Install a current Unsloth/Transformers combination that supports Qwen3.5; GPU memory requirements depend on model size, image count, and sequence length.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Generate narration

Extract the dataset first, then run commands from the repository root. Image paths in the released data are relative to this directory.

```bash
python src/generate.py \
  --data data/records/lectures_01.jsonl \
  --model unsloth/Qwen3.5-4B \
  --method zero_shot \
  --output outputs/generated.jsonl
```

Use `--unguided` for plain narration, `--method cot` for chain-of-thought prompting, or `--method few_shot --examples data/demonstrations.jsonl` for two demonstrations from the same subject category. Demonstrations must come from lectures distinct from the generation inputs. To use a trained adapter, set `--model checkpoints/paft --method train`.

## Fine-tuning

Fine-tuning uses only the ordered slide images as input. PAFT targets the annotated transcript; conventional SFT targets the plain transcript. Generation with `--method train` uses the same image-only input format.

Supply your training data in the JSONL format below. The release has no predefined experimental partition. Create lecture-disjoint partitions for your experiment, keeping all slides of a lecture together.

```bash
python src/paft.py \
  --data data/train.jsonl \
  --model unsloth/Qwen3.5-4B \
  --output checkpoints/paft
```

Use `src/train.py` for conventional SFT or `--model unsloth/Qwen3.5-9B` for the larger model. Defaults follow the source training configuration: LoRA rank/alpha 16, learning rate 0.0002, 10 epochs, batch size 3, gradient accumulation 6, and sequence length 30,000. Adjust batch size and sequence length to your hardware. This release saves epoch checkpoints and the final adapter.

## Collect source transcripts

The scraper follows the Downloads → Transcripts section of an NPTEL course page. It collects transcript links for the selected language and optionally downloads the linked PDFs.

```bash
pip install -r requirements-scraping.txt
python -m playwright install chromium
python src/scrape_nptel.py \
  --course-id 106106142 \
  --output collected/transcript_links.csv \
  --download-dir collected/pdfs
```

Omit `--download-dir` to collect links only. Repeat `--course-id` for multiple courses, or use `--courses courses.csv` with a numeric `course_id` column. Course-title identifiers beginning with `NOC:` must first be mapped to numeric NPTEL IDs. The default language is `English-Verified`; use `--language` for another available language.

The output CSV retains course and lecture identifiers, titles, language, source URLs, and downloaded file paths relative to `--download-dir`. Lecture numbers come from titles where available, otherwise from the displayed row order. Existing files are not overwritten. Collection stops on page or download failures; rows already written remain available in the output CSV. Source PDFs require slide extraction and transcript alignment before use with the training scripts.

## Prepare and annotate data

```bash
python src/prepare_data.py \
  --input data/lectures.csv \
  --image-root data/slides \
  --output data/lectures.jsonl
```

The converter accepts `course_id`, `lecture_number`, `category`, `slide_set`, `transcript_merged`, and optional `xml_train_annotated_transcript_clean` columns. `slide_set` is a serialized list. Use `--strip-prefix` to remove an existing image-root prefix, or the column options for different input schemas.

For new annotations, set `GEMINI_API_KEY` in your environment, install the sentence tokenizer, and provide your available Gemini model ID. This command sends transcript text to the Gemini API.

```bash
python -m nltk.downloader punkt_tab
python src/annotate.py \
  --data data/lectures.jsonl \
  --model YOUR_GEMINI_MODEL_ID \
  --output data/annotated.jsonl
```

Annotation operates on each slide block and preserves its text and position. Predictions should be reviewed before use as reference annotations.

## Data format

Each JSONL record contains:

| Field | Description |
|---|---|
| `id` | Unique record identifier |
| `course_id`, `lecture_number` | Original lecture identifiers |
| `category` | Subject category |
| `slide_set` | Ordered list of image paths |
| `transcript` | Original narration with normalized whitespace |
| `annotated_transcript` | Narration with paired pedagogical tags |


Full lecture records use `[NEXT_SLIDE]` between consecutive slide narrations. The nth narration block corresponds to the nth image in `slide_set`. All slides of each selected lecture are retained in their original order.

After extracting all 27 parts:

```text
data/
  selection.json
  lectures.csv
  records/
    lectures_01.jsonl
  lectures/
    lecture_0002/
      metadata.json
      transcript.txt
      annotated_transcript.txt
      slides/
        slide_001.png
        slide_002.png
        ...
    ...
    lecture_0942/
```

Both transcript versions retain the source wording with normalized whitespace. Annotation tags use `organization`, `explanation`, `illustration`, `question`, `contrast`, `conclusion`, and `non_verbal_cue`. The source spelling `non-verbal_cue` is normalized to `non_verbal_cue`.

The selection covers all 13 subject categories, varied lecture lengths and slide counts, and textual, mathematical, code, diagram, image, chart, and table content. Selection prioritizes balanced subject coverage, course diversity, pedagogical-tag diversity and moderate lecture lengths. See `data/selection.json` for the selected IDs and validation results. This is a reproducible structural selection, not a human-rated ranking. Records require complete image availability, nonempty slide narration, valid annotation boundaries, and preservation of the original narration. These checks do not establish a human-adjudicated annotation-accuracy score.

The training, generation, and annotation tools accept either a single JSONL file or a directory of JSONL files. Use `--data data/records` to load the entire release. The scripts are portable adaptations of the research code; this subset alone does not reproduce the paper's full experiments.

## Dataset overview

The release contains **100 complete lectures**, **1714 slide–transcript pairs**, and **79 courses**. Original lecture IDs are retained.

| Subject | Lectures |
|---|---:|
| Algorithms & Data Structures | 8 |
| Artificial Intelligence | 8 |
| Cloud Computing | 8 |
| Computer Graphics | 7 |
| Computer Networks | 8 |
| Computer Systems | 8 |
| Cybersecurity | 8 |
| Databases | 7 |
| Hardware & Embedded Systems | 7 |
| Human-Computer Interaction | 8 |
| Mathematics & Theory | 8 |
| Programming | 8 |
| Software Engineering | 7 |


## Upload and download

Upload all 27 ZIP files to `data/` and commit the changes. Git LFS is not required for these archives. The uncompressed dataset is below 2 GiB, but the ZIPs are larger than the individual extracted files.

The ZIP sizes are suitable for GitHub web uploads, but anonymous mirror downloads have not been verified. These parts exceed Anonymous GitHub's previously documented 8 MB per-file limit; do not assume that a successful GitHub upload makes them downloadable through the anonymous mirror.

## Acknowledgments

The source lectures, slides, and narration are from NPTEL and their respective instructors and institutions. Course identifiers and original slide attribution are retained.
