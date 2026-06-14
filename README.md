# scala-llm-fine-tuning

Fine-tune your LLM for Scala 3 – a practical plan and recipes for running on Apple Silicon (MacBook Air) using [MLX](https://github.com/ml-explore/mlx-lm).

---

## Table of Contents

1. [Goal](#goal)
2. [Phase 1 – Data Collection](#phase-1--data-collection)
3. [Phase 2 – Data Preparation & Formatting](#phase-2--data-preparation--formatting)
4. [Phase 3 – Fine-tuning with MLX on Apple Silicon](#phase-3--fine-tuning-with-mlx-on-apple-silicon)
5. [Phase 4 – Evaluation & Testing](#phase-4--evaluation--testing)
6. [Phase 5 – Automation for Future Scala Versions](#phase-5--automation-for-future-scala-versions)
7. [Community Patterns & References](#community-patterns--references)
8. [Implementation Roadmap](#implementation-roadmap)

---

## Goal

Produce a domain-specific LLM that understands and generates idiomatic **Scala 3** (Dotty) code, trained locally on a MacBook Air using Apple's [MLX](https://github.com/ml-explore/mlx-lm) framework, with a repeatable pipeline that can be updated whenever a new Scala 3 version ships.

---

## Phase 1 – Data Collection

### 1.1 Primary sources

| Source | Content | Notes |
|--------|---------|-------|
| [scala/scala3](https://github.com/scala/scala3) compiler repo | Compiler source, tests, examples – idiomatic Scala 3 | ~250 k lines of pure Scala 3 |
| [scala/scala3-examples](https://github.com/scala/scala3-examples) | Official Dotty example programs | Small but high-quality |
| [lampepfl/dotty-website](https://github.com/lampepfl/dotty-website) | Blog posts, migration guides | Prose + code snippets |
| [Scala 3 Book / Docs](https://docs.scala-lang.org/scala3/book/introduction.html) | Reference documentation | HTML scraped to Markdown |
| GitHub Code Search (`language:Scala`) | Real-world Scala 3 projects | Filter by `build.sbt` referencing `scalaVersion := "3.*"` |
| Scastie snippets ([scastie.scala-lang.org](https://scastie.scala-lang.org)) | Short, self-contained snippets | Public snippets via API |
| Stack Overflow / Scala tag | Q&A pairs | Use Stack Exchange Data Dump (CC BY-SA) |
| Coursera / Rock the JVM Scala 3 courses | Exercises and solutions | Check individual licences before use |

### 1.2 Filtering for Scala 3

When harvesting from GitHub or other sources, keep only files that are unambiguously Scala 3:

```bash
# files that use Scala 3-only syntax markers
grep -rl --include="*.scala" \
  -e "^enum " \
  -e "^given " \
  -e "^extension" \
  -e " derives " \
  -e "@main" \
  -e "transparent inline" \
  /path/to/corpus
```

Also filter `build.sbt` / `build.sc` to ensure `scalaVersion` starts with `3.`:

```bash
grep -rl 'scalaVersion.*:=.*"3\.' /path/to/repos
```

### 1.3 Licensing considerations

- Prefer **Apache 2.0**, **MIT**, **BSD**, or **CC BY-SA** licensed code.
- Document every source and its licence in `data/sources.csv`.
- Avoid GPL-only code unless you intend to publish under GPL.

---

## Phase 2 – Data Preparation & Formatting

### 2.1 MLX fine-tuning data format

MLX-LM LoRA fine-tuning expects **JSONL** files (`train.jsonl`, `valid.jsonl`, `test.jsonl`), where each line is one of:

**Instruction format** (recommended for a coding assistant):

```json
{"text": "<|im_start|>user\nWrite a Scala 3 function that reverses a list using tail recursion.\n<|im_end|>\n<|im_start|>assistant\ndef reverseList[A](list: List[A]): List[A] =\n  @annotation.tailrec\n  def loop(remaining: List[A], acc: List[A]): List[A] =\n    remaining match\n      case Nil          => acc\n      case head :: tail => loop(tail, head :: acc)\n  loop(list, Nil)\n<|im_end|>"}
```

**Completion format** (simpler, good for autocompletion):

```json
{"text": "// Scala 3 – tail-recursive list reversal\ndef reverseList[A](list: List[A]): List[A] =\n  @annotation.tailrec\n  def loop(remaining: List[A], acc: List[A]): List[A] =\n    remaining match\n      case Nil          => acc\n      case head :: tail => loop(tail, head :: acc)\n  loop(list, Nil)\n"}
```

Use the instruction format to build a **chat-style coding assistant**; use completion format for pure code generation.

### 2.2 Generating instruction pairs automatically

For code files extracted from open-source repos, generate instruction prompts synthetically:

1. **Template-based**: wrap each file or function in a fixed prompt template  
   (`"Explain what this Scala 3 code does and rewrite it with better variable names: …"`)
2. **LLM-assisted labelling**: use a large API model (GPT-4o, Claude 3.5) to generate a question for each snippet. Run offline/batch to save cost.
3. **Stack Overflow mining**: each SO question + accepted answer is already an instruction pair.

### 2.3 Data cleaning pipeline

```
raw_corpus/
├── github/          ← scraped .scala files
├── docs/            ← scraped HTML → Markdown
└── stackoverflow/   ← SEDE export

scripts/
├── 01_filter_scala3.py      ← keep Scala 3 files only
├── 02_extract_functions.py  ← split files into function-level chunks
├── 03_generate_prompts.py   ← wrap chunks in instruction template
├── 04_dedup.py              ← MinHash near-deduplication
├── 05_split.py              ← 90/5/5 train/valid/test split
└── 06_validate_jsonl.py     ← schema check + token count histogram
```

Recommended chunk size: **≤ 2 048 tokens** (fits most 7B base models with an 8 k context window).

### 2.4 Quality checks

- Remove files with syntax errors: `scalac -3 file.scala 2>&1 | grep -c error`
- Remove near-duplicates (MinHash / SimHash with Jaccard threshold 0.8)
- Ensure a balanced mix: library code, application code, test code, docs

---

## Phase 3 – Fine-tuning with MLX on Apple Silicon

### 3.1 Prerequisites

```bash
# Python 3.11+ on macOS with Apple Silicon
pip install mlx-lm

# Verify GPU is visible
python -c "import mlx.core as mx; print(mx.default_device())"
# → Device(gpu, 0)
```

### 3.2 Choose a base model

| Model | Size | Context | Notes |
|-------|------|---------|-------|
| `mlx-community/Mistral-7B-Instruct-v0.3-4bit` | ~4 GB | 32 k | Good general baseline |
| `mlx-community/CodeLlama-7b-Instruct-hf-4bit` | ~4 GB | 16 k | Code-focused |
| `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit` | ~4 GB | 128 k | Strong at code, long context |
| `mlx-community/deepseek-coder-6.7b-instruct-4bit` | ~3.5 GB | 16 k | Excellent code model |

Start with a **4-bit quantised** model so it fits in MacBook Air unified memory (8–16 GB).

### 3.3 LoRA fine-tuning

```bash
# Minimal working command
mlx_lm.lora \
  --model mlx-community/Qwen2.5-Coder-7B-Instruct-4bit \
  --train \
  --data data/processed/ \           # directory with train.jsonl, valid.jsonl
  --num-layers 16 \                  # LoRA adapter layers
  --batch-size 2 \
  --iters 1000 \
  --learning-rate 1e-4 \
  --save-every 100 \
  --adapter-path adapters/scala3-v1
```

Key MLX LoRA flags:

| Flag | Recommended value | Reason |
|------|-------------------|--------|
| `--num-layers` | 16–32 | More layers → higher quality, more memory |
| `--batch-size` | 2–4 | MacBook Air 16 GB limit |
| `--grad-checkpoint` | on | Reduces peak memory by 40% |
| `--lora-rank` | 8–16 | LoRA rank; higher = more parameters |
| `--iters` | 1 000–5 000 | Monitor validation loss to avoid overfitting |

### 3.4 Merge & test

```bash
# Fuse LoRA weights into the base model
mlx_lm.fuse \
  --model mlx-community/Qwen2.5-Coder-7B-Instruct-4bit \
  --adapter-path adapters/scala3-v1 \
  --save-path models/scala3-coder-v1

# Quick sanity test
mlx_lm.generate \
  --model models/scala3-coder-v1 \
  --prompt "Write a Scala 3 given instance for the Ordering typeclass for a case class Person(name: String, age: Int)."
```

---

## Phase 4 – Evaluation & Testing

### 4.1 Automated test set

Create a **held-out test set** (`test.jsonl`) of hand-written Scala 3 tasks with reference solutions:

```
data/
└── test_tasks/
    ├── 001_tail_recursion.json
    ├── 002_given_instances.json
    ├── 003_extension_methods.json
    ├── 004_enums_adt.json
    ├── 005_type_classes.json
    ├── 006_opaque_types.json
    ├── 007_context_functions.json
    └── ...
```

Each task file:

```json
{
  "id": "003",
  "prompt": "Write a Scala 3 extension method `words` on String that splits a sentence into a List[String] of words.",
  "reference": "extension (s: String)\n  def words: List[String] = s.split(\"\\\\s+\").toList",
  "tags": ["extension-methods", "stdlib"]
}
```

### 4.2 Compilation-based evaluation

The strongest signal for code generation quality is **whether the code compiles**:

```bash
# scripts/eval_compile.sh
mlx_lm.generate --model models/scala3-coder-v1 --prompt "$PROMPT" > /tmp/output.scala
scalac -3 /tmp/output.scala 2>&1
echo "Exit code: $?"
```

Automate over the full test set and report **pass@1** and **pass@k** (generate k candidates, pass if any compile).

### 4.3 Semantic / unit-test evaluation

For tasks that have a known expected output, embed a `main` method and run:

```bash
scala-cli run /tmp/output.scala -- | diff - expected_output.txt
```

### 4.4 Human evaluation rubric

| Criterion | Weight |
|-----------|--------|
| Compiles without errors | 40% |
| Follows Scala 3 idioms (given/using, enums, extensions) | 25% |
| Passes unit tests | 20% |
| Code readability & naming | 15% |

---

## Phase 5 – Automation for Future Scala Versions

### 5.1 Version-aware corpus pipeline

Store all data with a `scala_version` tag:

```json
{"text": "...", "meta": {"source": "github", "repo": "acme/myapp", "scala_version": "3.5.0", "license": "Apache-2.0"}}
```

When a new Scala 3 patch/minor version is released:

1. **Run the scraper** targeting repos that have bumped their `scalaVersion`.
2. **Re-run the cleaning pipeline** – new code is automatically deduplicated against the existing corpus.
3. **Incrementally fine-tune** the existing adapter on the delta dataset (fewer iterations needed).

### 5.2 GitHub Actions workflow (future)

```yaml
# .github/workflows/refresh-corpus.yml
name: Refresh Scala 3 corpus
on:
  schedule:
    - cron: '0 3 1 * *'   # first day of each month
  workflow_dispatch:

jobs:
  scrape:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Scrape new Scala 3 repos
        run: python scripts/01_filter_scala3.py --since last-run
      - name: Deduplicate & split
        run: python scripts/04_dedup.py && python scripts/05_split.py
      - name: Upload to artefact store
        run: ...
```

### 5.3 Tracking Scala 3 release cycle

Subscribe to:
- [scala-lang.org/blog](https://www.scala-lang.org/blog/) — release announcements
- [github.com/scala/scala3/releases](https://github.com/scala/scala3/releases) — GitHub release feed (RSS available)
- [users.scala-lang.org](https://users.scala-lang.org) — community discussion

When a new version introduces syntax changes (e.g., new soft-keywords, new stdlib additions), update the syntax-filter patterns in `scripts/01_filter_scala3.py` accordingly.

---

## Community Patterns & References

### Existing community work

| Project | What it shows |
|---------|--------------|
| [microsoft/humaneval](https://github.com/openai/human-eval) | Gold-standard code eval benchmark; adapt tasks to Scala 3 |
| [bigcode-project/starcoder](https://github.com/bigcode-project/starcoder) | Large-scale open code model; use as a base or comparison |
| [mlx-lm LoRA examples](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm) | Official Apple MLX fine-tuning recipes |
| [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) | Community fine-tuning framework (not MLX, but widely used patterns) |
| [LLM.int8() / QLoRA paper](https://arxiv.org/abs/2305.14314) | Theoretical foundation for quantised LoRA |

### Key insight from the community

> The most impactful factor for domain-specific code fine-tuning is **data quality over quantity**. A curated set of 10 000 high-quality Scala 3 instruction pairs typically outperforms 100 000 noisy pairs scraped from the web.

---

## Implementation Roadmap

```
Week 1  Data collection
         ├── Set up GitHub scraper (PyGitHub / ghapi)
         ├── Scrape scala/scala3 + top-100 Scala 3 repos by stars
         └── Download Stack Overflow Scala data dump

Week 2  Data pipeline
         ├── Implement scripts/01–06
         ├── Generate ~10 000 instruction pairs
         └── Review 200 random samples manually

Week 3  Baseline fine-tune
         ├── Install mlx-lm, download base model
         ├── Run first fine-tuning (1 000 iters, small dataset)
         └── Evaluate: compilation rate + subjective quality

Week 4  Iterate & evaluate
         ├── Build test_tasks/ benchmark (50 tasks)
         ├── Tune hyperparameters (rank, layers, LR)
         └── Compare adapters on benchmark

Week 5+ Automation & productionisation
         ├── GitHub Actions corpus refresh workflow
         ├── Model versioning (adapter v1, v2, …)
         └── Optional: publish adapter to Hugging Face Hub
```

---

## Directory Structure (target)

```
scala-llm-fine-tuning/
├── README.md                  ← this file
├── data/
│   ├── raw/                   ← unprocessed scraped files (gitignored)
│   ├── processed/
│   │   ├── train.jsonl
│   │   ├── valid.jsonl
│   │   └── test.jsonl
│   ├── test_tasks/            ← hand-written evaluation tasks
│   └── sources.csv            ← licence tracking
├── scripts/
│   ├── 01_filter_scala3.py
│   ├── 02_extract_functions.py
│   ├── 03_generate_prompts.py
│   ├── 04_dedup.py
│   ├── 05_split.py
│   └── 06_validate_jsonl.py
├── adapters/                  ← LoRA adapter checkpoints (gitignored)
├── models/                    ← fused models (gitignored)
└── .github/
    └── workflows/
        └── refresh-corpus.yml
```

---

## Quick-start checklist

- [ ] Install `mlx-lm`: `pip install mlx-lm`
- [ ] Choose and download a base model (see §3.2)
- [ ] Run `scripts/01_filter_scala3.py` to collect an initial corpus
- [ ] Run `scripts/02–05` to build `train.jsonl` / `valid.jsonl`
- [ ] Run `mlx_lm.lora` for the first fine-tuning run
- [ ] Evaluate using `scripts/eval_compile.sh` and `data/test_tasks/`
- [ ] Iterate!

---

*This plan is intended as a living document. Open a PR to suggest improvements or share your results.*
