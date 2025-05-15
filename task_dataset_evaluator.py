
import torch
import numpy as np
from PIL import Image
from open_clip import create_model_from_pretrained, get_tokenizer

def evaluate_sample(sample, model, tokenizer, preprocess, device, context_length=256):
    q_type = sample['q_type']
    image_paths = sample['image_paths']
    text_inputs = sample['text_inputs']
    correct_index = sample['correct_index']

    try:
        images = [preprocess(Image.open(p).convert("RGB")) for p in image_paths]
    except Exception as e:
        print(f"Error loading image(s) for QA ID {sample['qa_id']}: {e}")
        return None

    image_batch = torch.stack(images).to(device)
    texts = tokenizer(text_inputs, context_length=context_length).to(device)

    with torch.no_grad():
        if q_type in ["img2txt", "img2size", "img2attrib"]:
            image_features, text_features, logit_scale = model(image_batch[0].unsqueeze(0), texts)
            logits = (logit_scale * image_features @ text_features.T).softmax(dim=-1)
            pred_index = logits.argmax().item()

        elif q_type in ["ct2txt", "ct2attrib"]:
            num_slices = image_batch.shape[0]
            center_idx = num_slices // 2
            distances = np.array([i - center_idx for i in range(num_slices)])
            gauss_weights = np.exp(-0.5 * (distances / 1.0) ** 2)
            gauss_weights /= gauss_weights.sum()

            all_image_features = []
            for i in range(num_slices):
                img = image_batch[i].unsqueeze(0)
                features, _, _ = model.encode_image(img)
                all_image_features.append(features.squeeze(0) * gauss_weights[i])

            weighted_image_feature = torch.stack(all_image_features).sum(dim=0).unsqueeze(0)
            text_features = model.encode_text(texts)
            logit_scale = model.logit_scale.exp()
            logits = (logit_scale * weighted_image_feature @ text_features.T).softmax(dim=-1)
            pred_index = logits.argmax().item()

        elif q_type in ["txt2img", "txt2bbox"]:
            image_features, text_features, logit_scale = model(image_batch, texts)
            logits = (logit_scale * image_features @ text_features.T).softmax(dim=0)
            pred_index = logits.argmax().item()

        else:
            print(f"Unknown q_type: {q_type}")
            return None

    return {
        "qa_id": sample['qa_id'],
        "q_type": q_type,
        "predicted_index": pred_index,
        "correct_index": correct_index,
        "is_correct": pred_index == correct_index
    }

def run_evaluation(dataset, context_length=256):
    model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    results = []
    task_stats = {}

    for sample in dataset:
        result = evaluate_sample(sample, model, tokenizer, preprocess, device, context_length)
        if result is None:
            continue
        results.append(result)
        q_type = result['q_type']
        task_stats.setdefault(q_type, {"correct": 0, "total": 0})
        task_stats[q_type]["total"] += 1
        if result['is_correct']:
            task_stats[q_type]["correct"] += 1

    return results, task_stats
