import os
import json
import argparse
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from collections import Counter
from bert_score import score as bertscore
from sentence_transformers import SentenceTransformer, util

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Tokenizer with fallback
def safe_tokenize(text):
    try:
        return word_tokenize(text)
    except Exception as e:
        print(f"[Tokenizer Warning] Fallback to .split(): {e}")
        return text.split()

# ROUGE-1 scorer
def rouge_n_score(reference, candidate, n=1):
    def ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    ref_ngrams = ngrams(reference.split(), n)
    cand_ngrams = ngrams(candidate.split(), n)
    ref_count = Counter(ref_ngrams)
    cand_count = Counter(cand_ngrams)
    overlap = sum((ref_count & cand_count).values())
    total_cand = len(cand_ngrams)
    total_ref = len(ref_ngrams)
    recall = overlap / total_ref if total_ref > 0 else 0
    precision = overlap / total_cand if total_cand > 0 else 0
    f_score = 2 * recall * precision / (recall + precision) if recall + precision > 0 else 0
    return f_score

# Load SentenceTransformer for cosine similarity
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Main scoring function
def calculate_all_scores(ground_truths, predictions):
    bleu_scores, meteor_scores, rouge_scores = [], [], []
    smoothie = SmoothingFunction().method4

    # Compute cosine similarity
    embeddings_ref = embedder.encode(ground_truths, convert_to_tensor=True)
    embeddings_pred = embedder.encode(predictions, convert_to_tensor=True)
    cosine_scores = util.cos_sim(embeddings_pred, embeddings_ref).diagonal()

    # Compute BERTScore
    _, _, F1 = bertscore(predictions, ground_truths, lang="en", verbose=False)

    for gt, pred in zip(ground_truths, predictions):
        reference = safe_tokenize(gt)
        candidate = safe_tokenize(pred)

        bleu = sentence_bleu([reference], candidate, smoothing_function=smoothie, weights=(1, 0, 0, 0))
        meteor = meteor_score([reference], candidate)
        rouge = rouge_n_score(gt, pred, n=1)

        bleu_scores.append(bleu)
        meteor_scores.append(meteor)
        rouge_scores.append(rouge)

    return {
        "BLEU-1": sum(bleu_scores) / len(bleu_scores),
        "METEOR": sum(meteor_scores) / len(meteor_scores),
        "ROUGE-1": sum(rouge_scores) / len(rouge_scores),
        "BERTScore": F1.mean().item(),
        "CosineSim": cosine_scores.mean().item()
    }

# Evaluate a caption JSON file
def evaluate_caption_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    predictions = [entry["caption"].strip() for entry in data]
    ground_truths = [entry["ground_truth"].strip() for entry in data]

    print(f"🔍 Evaluating {len(predictions)} samples from {json_path} ...")
    return calculate_all_scores(ground_truths, predictions)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_file", required=True, help="Path to GPT-4V or Gemini caption result JSON file")
    args = parser.parse_args()

    scores = evaluate_caption_json(args.json_file)

    print("\n--- Caption Evaluation Results ---")
    print(f"{'BLEU-1':<12}: {scores['BLEU-1']:.4f}")
    print(f"{'METEOR':<12}: {scores['METEOR']:.4f}")
    print(f"{'ROUGE-1':<12}: {scores['ROUGE-1']:.4f}")
    print(f"{'BERTScore':<12}: {scores['BERTScore']:.4f}")
    print(f"{'CosineSim':<12}: {scores['CosineSim']:.4f}")
