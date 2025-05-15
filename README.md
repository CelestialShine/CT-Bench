# 🧠 CT-Bench: Benchmarking Multimodal AI in Computed Tomography

This repository contains the official code for our NeurIPS 2025 Datasets & Benchmarks (DB) Track submission:

**CT-Bench: A Comprehensive Benchmark for Multimodal AI in Computed Tomography Analysis**

CT-Bench supports multimodal lesion-level CT analysis with:
- 🧩 20,335 annotated lesions (bounding boxes, size, descriptions)
- 🧠 2,850 expert-verified QA tasks spanning 7 clinical reasoning tasks
- ⚙️ Baselines with GPT-4V, BiomedCLIP, and fine-tuned models

---

## 📦 Dataset

The dataset includes:
- **Lesion Image & Metadata Set**: lesion slices (with/without BBox), metadata (size, description)
- **QA Benchmark Component**: VQA tasks in both LLM and CLIP styles, with hard negatives

📝 **Download instructions and access links are included in the NeurIPS submission.**

### Folder Structure After Extraction

```
data/
├── clip/              # QA pairs for CLIP-style models
├── llm/               # QA pairs for LLM-style models
├── lesion_bbox/       # PNG images with bounding boxes (cropped)
├── lesion_nobox/      # PNG images without bounding boxes
├── Metadata.csv       # Lesion info: id, description, size, split
├── qa_clip.json       # Structured QA for CLIP models
├── qa_llm.json        # Structured QA for LLM models
```

---

## 🗂️ Project Structure

```
ct-bench/
├── captioning/                    # Image captioning with GPT-4V or others
│   └── image_caption.py
├── training/                      # Fine-tuning BiomedCLIP, RadFM, etc.
│   └── biomedclip.py
├── qa_llm/                        # QA evaluation using LLM-style models
│   └── run_qa_vlm.py
├── qa_clip/                       # QA evaluation using CLIP-style models
│   └── run_qa_clip.py
├── models/                        # Modular model wrappers
│   ├── __init__.py
│   ├── base.py
│   ├── gpt4v.py
│   ├── biomedclip.py
├── data/                          # External dataset (see above)
├── results/                       # Output logs, captions, predictions
├── configs/                       # Optional config files
├── requirements.txt
└── README.md
```

---

## 🖼️ 1. Lesion Image Captioning with GPT-4V

Generate CT lesion descriptions using GPT-4V.

```bash
python captioning/image_caption.py \
  --image_dir data/lesion_nobox \
  --metadata data/Metadata.csv \
  --output_dir results/captions
```

- Input: `.png` images, `Metadata.csv`
- Output: `gpt4v_captions.json` with model-generated and ground-truth captions
- Requires `OPENAI_API_KEY` to be set in your environment

---

## 🧪 2. Fine-Tuning BiomedCLIP on CT-Bench

Train contrastive models using bounding box lesion slices.

```bash
python training/biomedclip.py \
  --train --input_dir data/lesion_bbox \
  --metadata data/Metadata.csv \
  --output_dir results/biomedclip
```

Supports both with and without BBox supervision:
- Use `--with_bbox` flag to include cropped regions
- Logs and model checkpoints saved to `results/biomedclip/`

---

## 🤖 3. QA Benchmark: LLM-Style (GPT-4V, Gemini, etc.)

Evaluate large language-vision models on clinical lesion QA tasks.

```bash
python qa_llm/run_qa_vlm.py \
  --model gpt4v \
  --qa_file data/qa_llm.json \
  --input_dir data/llm/test
```

Supported tasks:
- `img2txt`, `ct2txt`, `img2attrib`, `ct2attrib`, `img2size`, etc.

---

## 🎯 4. QA Benchmark: CLIP-Style (BiomedCLIP, PMC-CLIP)

Use image-text retrieval to answer QA tasks with hard negatives.

```bash
python qa_clip/run_qa_clip.py \
  --model biomedclip \
  --qa_file data/qa_clip.json \
  --input_dir data/clip/test
```

Tasks include:
- `txt2img`, `txt2bbox`, `img2attrib`

---

## ⚙️ Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

Includes:
- Python 3.8+
- PyTorch 2.0+
- OpenAI SDK
- Pillow, TQDM, etc.

---

## 🔒 Reproducibility & Anonymity

- This repository is anonymized for NeurIPS review.
- Dataset access and additional models (e.g., Gemini, Dragonfly) are described in the paper.
- Only GPT-4V and BiomedCLIP are exposed for simplicity and reproducibility.

---

## 📬 Contact

Please refer to the NeurIPS submission for correspondence and dataset request info.
