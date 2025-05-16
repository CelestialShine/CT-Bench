# NOTE:
# This script provides a unified interface for LLM-style QA benchmarking.
# For models like RadFM, LLaVA-Med, or Dragonfly, whose architectures and input handling
# differ significantly, we use mock logic here. Actual integration should be implemented
# inside their respective model wrappers.
# You can use this script as a base to extract image paths, questions, and choices
# to feed into those models separately.

import os
import json
import base64
import time
import argparse
from mimetypes import guess_type
import requests
import google.generativeai as genai

# Helper: encode image to base64 data URL
def local_image_to_data_url(image_path):
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:{mime_type};base64,{base64_encoded_data}"

# GPT-4V call
def call_gpt4v(image_data_url, question, choices, api_key, endpoint):
    headers = {
        "api-key": api_key,
        "Content-Type": "application/json"
    }
    options_text = "\n".join([f"{key}: {value}" for key, value in choices.items()])
    prompt = (
        f"{question}\nOptions:\n{options_text}\n"
        "Please provide the output in the following format:\n"
        "{\n"
        "  \"selected_option\": \"<selected_option>\",\n"
        "  \"explanation\": \"<explanation>\"\n"
        "}"
    )
    payload = {
        "messages": [
            {"role": "system", "content": "Imagine you are a radiologist. Provide a clinical answer."},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_data_url}}
            ]}
        ],
        "max_tokens": 1000,
        "top_p": 1,
        "temperature": 0,
    }
    response = requests.post(endpoint, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()['choices'][0]['message']['content']

# Gemini call
def call_gemini(image_path, question, choices):
    img = open(image_path, "rb").read()
    model = genai.GenerativeModel("gemini-1.5-flash")
    options_text = "\n".join([f"{key}: {value}" for key, value in choices.items()])
    prompt = (
        f"{question}\nOptions:\n{options_text}\n"
        "Please provide the output in the following format:\n"
        "{\n"
        "  \"selected_option\": \"<selected_option>\",\n"
        "  \"explanation\": \"<explanation>\"\n"
        "}"
    )
    response = model.generate_content([prompt, img])
    response.resolve()
    return response.text

# Main run function
def run_qa_all_models(args):
    with open(args.qa_file, "r") as f:
        data = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    results = []

    for i, entry in enumerate(data):
        image_path = os.path.join(args.image_dir, entry["image_path"])
        if not os.path.exists(image_path):
            print(f"Skipping missing image: {image_path}")
            continue
        try:
            if args.model == "gpt4v":
                image_data_url = local_image_to_data_url(image_path)
                raw_response = call_gpt4v(image_data_url, entry["question"], entry["choices"], args.api_key, args.endpoint)
            elif args.model == "gemini":
                raw_response = call_gemini(image_path, entry["question"], entry["choices"])
            elif args.model == "radfm":
                print(f"[RadFM MOCK] Processing: {entry['lesion_idx']}")
                selected_option = "A"
                explanation = "Simulated RadFM output."
                answer = {"selected_option": selected_option, "explanation": explanation}
            elif args.model == "llava":
                print(f"[LLaVA-Med MOCK] Processing: {entry['lesion_idx']}")
                selected_option = "B"
                explanation = "Simulated LLaVA-Med output."
                answer = {"selected_option": selected_option, "explanation": explanation}
            elif args.model == "dragonfly":
                print(f"[Dragonfly MOCK] Processing: {entry['lesion_idx']}")
                selected_option = "C"
                explanation = "Simulated Dragonfly output."
                answer = {"selected_option": selected_option, "explanation": explanation}
            else:
                raise ValueError("Unsupported model: " + args.model)

            raw_response = raw_response.strip().lstrip("```json").rstrip("```").strip()
            try:
                answer = json.loads(raw_response)
            except:
                answer = {"selected_option": None, "explanation": raw_response}

            result = {
                "i": i,
                "lesion_idx": entry["lesion_idx"],
                "q_type": entry["q_type"],
                "question": entry["question"],
                "choices": entry["choices"],
                "correct_choice": entry["correct_choice"],
                "answer": answer
            }

            results.append(result)
            with open(os.path.join(args.output_dir, f"{i}_output.json"), "w") as f:
                json.dump(result, f, indent=2)

            print(f"[{i+1}/{len(data)}] Done: {entry['lesion_idx']}")

        except Exception as e:
            print(f"❌ Error at index {i}: {e}")
            continue

        time.sleep(args.sleep)

    with open(os.path.join(args.output_dir, "output_data.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Finished. Results saved to: {args.output_dir}")

# Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa_file", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model", type=str, choices=["gpt4v", "gemini", "radfm", "llava", "dragonfly"], default="gpt4v")
    parser.add_argument("--api_key", type=str, default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--endpoint", type=str, default="https://bionlp-west.openai.azure.com/openai/deployments/gpt-4-vision/chat/completions?api-version=2024-02-15-preview")
    parser.add_argument("--sleep", type=int, default=5)
    args = parser.parse_args()
    run_qa_all_models(args)
