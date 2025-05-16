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

## 🗂️ Project Structure

```
ct-bench/
├── captioning/                    # Image captioning with GPT-4V or Gemini
│   └── caption_generator.py
├── training/                      # Fine-tuning BiomedCLIP, RadFM, etc.
│   └── biomedclip.py
├── qa_llm/                        # QA evaluation using LLM-style models
│   └── run_qa_vlm.py
├── qa_clip/                       # QA evaluation using CLIP-style models
│   └── run_qa_clip.py
├── evaluation/                    # Metric evaluation for caption outputs
│   └── evaluate_caption_json.py
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

---

## 🧪 2. Fine-Tuning BiomedCLIP on CT-Bench

Train contrastive models using bounding box lesion slices.

```bash
python fine_tune_biomedclip_metadata_arg.py --image_dir data/lesion_nobox
```

You can switch it to lesion_bbox/

---

## 🤖 3. QA Benchmark: LLM-Style (GPT-4V, Gemini, etc.)

Evaluate large vision-language models on lesion-level multiple choice tasks.

```bash
python run_llm_qa_all_models_notice.py \
  --model gpt4v \
  --qa_file data/qa_llm.json \
  --input_dir data/llm/test \
  --output_dir results/qa_gpt4v/
```

Tasks: `img2txt`, `ct2txt`, `img2attrib`, `ct2attrib`, `img2size`, etc.
Each question is saved in output_data.json with model answers. GPT-4V and Gemini include explanations. Other models return answers only.
---

## 🎯 4. QA Benchmark: CLIP-Style (BiomedCLIP, PMC-CLIP)

Use image-text similarity for retrieval-based multiple choice.

```bash
python qa_clip/run_qa_clip.py \
  --model biomedclip \
  --qa_file data/qa_clip.json \
  --input_dir data/clip/test
```

Tasks: `txt2img`, `txt2bbox`, `img2attrib`

---

## 📊 5. Evaluation of Captions

Evaluate Gemini or GPT-4V caption output using automatic metrics:

```bash
python evaluation/evaluate_caption_json.py \
  --json_file results/captions/gpt4v_captions.json
```

Metrics:
- BLEU-1, METEOR, ROUGE-1
- BERTScore, Cosine Similarity (SBERT)

---

## 🧪 Tested Models

CT-Bench was used to evaluate a diverse set of models across both captioning and QA tasks. Model settings and implementation details are available at the links below:

| Model              | Type                   | Source / Settings |
|--------------------|------------------------|--------------------|
| **BiomedCLIP**       | CLIP-style (medical)     | [HuggingFace](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) |
| **RadFM**            | LLM-style (medical VLM)  | [GitHub](https://github.com/chaoyi-wu/RadFM/tree/main/src) |
| **Dragonfly**        | General LLM + vision     | [Together AI](https://www.together.ai/blog/dragonfly-v1) |
| **PMC-CLIP**         | CLIP-style (medical)     | [GitHub](https://github.com/WeixiongLin/PMC-CLIP/tree/b0b81e3629740b4af837338ab5afa46e5d03a18e) |
| **LLaVA-Med**        | Open-source medical VLM  | [GitHub](https://github.com/microsoft/LLaVA-Med) |
| **GPT-4V**           | Commercial LLM-Vision API| [Azure OpenAI Service](https://azure.microsoft.com/en-us/products/ai-services/openai-service) |
| **Gemini 1.5 Pro**   | Commercial LLM-Vision API| [Google Gemini](https://ai.google.dev/gemini-api) |

Model-specific configurations, input styles (prompting or retrieval), and evaluation setups follow the guidelines in their respective repositories.

---

## ⚙️ Requirements

Install core dependencies with:

```bash
pip install -r requirements.txt
```

This includes:
- `pandas`, `nltk`, `tqdm`, `Pillow`
- `sentence-transformers` for cosine similarity
- `bert-score` for semantic caption evaluation

> 💡 Note: Each model (e.g., BiomedCLIP, GPT-4V, Gemini) may require additional packages. Refer to the respective model documentation or script headers for setup.


## 🔒 Reproducibility 

- This repository is anonymized for NeurIPS review.
- GPT-4V and BiomedCLIP are primary baselines.
- Other model results (Gemini, Dragonfly, etc.) are reproducible using included tools.

---

## 📬 Contact

Please refer to the NeurIPS submission for correspondence and dataset access instructions.
