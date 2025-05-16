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

📝 **Download links are included in the NeurIPS submission.**

### Folder Structure After Extraction

```
data/
├── clip/              # QA pairs for CLIP-style models
├── llm/               # QA pairs for LLM-style models
├── lesion_bbox/       # PNG images with bounding boxes 
├── lesion_nobox/      # PNG images without bounding boxes
├── Metadata.csv       # Lesion info: id, description, size, split
├── qa_clip.json       # Structured QA for CLIP models
├── qa_llm.json        # Structured QA for LLM models
```

---

# 📁 Project Structure

```
.
├── caption/
│   ├── caption_generator.py          # Script for generating captions
│   └── evaluate_caption_json.py      # Evaluates caption output in JSON format
│
├── qa_clip/
│   ├── run_all_tasks.py              # Runs all QA tasks using CLIP
│   ├── task_dataset_evaluator.py     # Evaluates QA datasets
│   └── task_dataset_wrapper.py       # Wraps datasets for CLIP QA
│
├── qa_llm/
│   └── run_llm.py                    # Script to run LLM-based QA
│
├── training/
│   └── fine_tune.py  # Fine-tuning script for BioMedCLIP
│
└── requirements.txt                 # Python package dependencies
```


---

## 🖼️ 1. Lesion Image Captioning

Generate CT lesion descriptions using GPT-4V or Gemini:

```bash
python captioning/caption_generator.py \
  --model gpt4v \
  --image_dir data/lesion_bbox \
  --metadata data/Metadata.csv \
  --output_dir results/captions
```

Change `--model gemini` to use Gemini instead.

- Input: `.png` images, `Metadata.csv`
- Output: `gpt4v_captions.json` or `gemini_captions.json`
- Requires `OPENAI_API_KEY` or `GOOGLE_API_KEY` to be set

Evaluate captioning performance:

```bash
python captioning/evaluate_caption_json.py \
  --json_file results/captions/gpt4v_captions.json
```

Metrics:
- BLEU-1, METEOR, ROUGE-1
- BERTScore
- SBERT Cosine Similarity

---

## 🧪 2. Fine-Tuning on CT-Bench

Train contrastive models using bounding box or no-box CT slices.

```bash
python training/fine_tune.py \
  --image_dir data/lesion_nobox
```



---

## 🤖 3. QA Benchmark: LLM-Style (GPT-4V, Gemini, etc.)

Run clinical lesion-level QA tasks in an LLM-style prompt format:

```bash
python qa_llm/run_llm.py \
  --model gpt4v \
  --qa_file data/qa_llm.json \
  --input_dir data/llm/test \
  --output_dir results/qa_gpt4v/
```

- Tasks: `img2txt`, `ct2txt`, `img2attrib`, `ct2attrib`, `img2size`, etc.
- GPT-4V and Gemini output answers with explanations.
- Other models (e.g. LLaVA-Med, RadFM) only return predicted choice (A/B/C/D).

> ⚠️ Other models are integrated manually inside their repo; this script provides only task I/O and metadata handling.

---

## 🎯 4. QA Benchmark: CLIP-Style (BiomedCLIP, PMC-CLIP)

Run contrastive QA using image-text similarity:

```bash
cd qa_clip
python run_all_tasks.py
```

In `run_all_tasks.py`, set:
- `dataset_path = "qa_clip_all_final_corrected.json"`
- `image_base_path = ""` (adjust to image folder)
- `output_file = "biomedclip_eval_results.json"`

Each task is evaluated independently. Accuracy is printed per task.

- Tasks: `txt2img`, `txt2bbox`, `img2attrib`, `ct2txt`, `ct2attrib`, etc.

---



## 🧪 Tested Models

| Model              | Type                   | Source / Settings |
|--------------------|------------------------|--------------------|
| **BiomedCLIP**       | CLIP-style (medical)     | [HuggingFace](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) |
| **RadFM**            | LLM-style (medical VLM)  | [GitHub](https://github.com/chaoyi-wu/RadFM/tree/main/src) |
| **Dragonfly**        | General LLM + vision     | [Together AI](https://www.together.ai/blog/dragonfly-v1) |
| **PMC-CLIP**         | CLIP-style (medical)     | [GitHub](https://github.com/WeixiongLin/PMC-CLIP/tree/b0b81e3629740b4af837338ab5afa46e5d03a18e) |
| **LLaVA-Med**        | Open-source medical VLM  | [GitHub](https://github.com/microsoft/LLaVA-Med) |
| **GPT-4V**           | Commercial LLM-Vision API| [Azure OpenAI Service](https://azure.microsoft.com/en-us/products/ai-services/openai-service) |
| **Gemini 1.5 Pro**   | Commercial LLM-Vision API| [Google Gemini](https://ai.google.dev/gemini-api) |

> ⚠️ Model-specific code and settings (e.g., RadFM, LLaVA-Med) are not included in this repo due to complexity. We provide example wrappers and I/O only.

---

## ⚙️ Requirements

Install core dependencies with:

```bash
pip install -r requirements.txt
```

Includes:
- `pandas`, `nltk`, `tqdm`, `Pillow`
- `sentence-transformers`
- `bert-score`

> 💡 Additional model packages (e.g. `openai`, `open-clip-torch`, etc.) should be installed as needed per script or model.

---

## 🔒 Reproducibility

- This repository is anonymized for NeurIPS review.
- GPT-4V and BiomedCLIP are the main benchmarks.
- Gemini, RadFM, and other models are tested using the same dataset and task formats.

---

## 📬 Contact

Please refer to the NeurIPS submission for correspondence and dataset access instructions.
