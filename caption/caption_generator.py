import os
import csv
import json
import base64
import argparse
from PIL import Image
from tqdm import tqdm

# Optional: Gemini
try:
    import google.generativeai as genai
except ImportError:
    genai = None

# Optional: GPT-4V (OpenAI)
try:
    import openai
except ImportError:
    openai = None

def encode_image_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def generate_gpt4v_caption(image_path, lesion_id):
    base64_image = encode_image_base64(image_path)
    try:
        response = openai.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Describe this CT lesion image in radiology terms. Lesion ID: {lesion_id}"
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                        }
                    ]
                }
            ],
            max_tokens=256
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[GPT-4V error: {e}]"

def generate_gemini_caption(image_path, lesion_id):
    try:
        img = Image.open(image_path)
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content([
            {
                'role': 'user',
                'parts': [
                    f"This image is with a bounding box created by a radiologist. "
                    f"Generate a short radiological impression based on this image. Lesion ID: {lesion_id}",
                    img
                ]
            }
        ])
        response.resolve()
        return response.text.strip()
    except Exception as e:
        return f"[Gemini error: {e}]"

def run_captioning(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # Load metadata
    with open(args.metadata, newline="") as f:
        reader = csv.DictReader(f)
        metadata = list(reader)

    results = []

    for row in tqdm(metadata, desc=f"Captioning with {args.model.upper()}"):
        lesion_id = row["lesion_idx"]
        image_path = os.path.join(args.image_dir, f"{lesion_id}.png")
        if not os.path.exists(image_path):
            continue

        if args.model == "gpt4v":
            caption = generate_gpt4v_caption(image_path, lesion_id)
        elif args.model == "gemini":
            caption = generate_gemini_caption(image_path, lesion_id)
        else:
            raise ValueError("Model must be 'gpt4v' or 'gemini'")

        results.append({
            "lesion_id": lesion_id,
            "caption": caption,
            "ground_truth": row["description"],
            "size": row.get("size", ""),
            "model": args.model
        })

    # Save output
    output_file = os.path.join(args.output_dir, f"{args.model}_captions.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Captions saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["gpt4v", "gemini"], required=True, help="Choose captioning model")
    parser.add_argument("--image_dir", required=True, help="Path to .png images")
    parser.add_argument("--metadata", required=True, help="Path to Metadata.csv")
    parser.add_argument("--output_dir", default="results/captions", help="Output directory for JSON results")
    args = parser.parse_args()

    if args.model == "gpt4v":
        if "OPENAI_API_KEY" not in os.environ:
            raise EnvironmentError("Set OPENAI_API_KEY environment variable.")
        openai.api_key = os.getenv("OPENAI_API_KEY")
    elif args.model == "gemini":
        if genai is None:
            raise ImportError("Google Generative AI SDK is not installed.")
        if "GOOGLE_API_KEY" not in os.environ:
            raise EnvironmentError("Set GOOGLE_API_KEY environment variable.")
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

    run_captioning(args)
