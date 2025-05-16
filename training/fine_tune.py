
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from PIL import Image
from open_clip import create_model_from_pretrained, get_tokenizer
import os
import pandas as pd

# Load metadata
csv_file_path = 'Metadata.csv'
df = pd.read_csv(csv_file_path)

# Split into train/test
train_df = df[df['dataset'] == 'train']
eval_df = df[df['dataset'] == 'test']

images = train_df['lesion_idx'].tolist()
captions = train_df['description'].tolist()
eval_images = eval_df['lesion_idx'].tolist()
eval_captions = eval_df['description'].tolist()

# Load BiomedCLIP model and tokenizer
model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Custom Dataset
class LesionDataset(Dataset):
    def __init__(self, images, captions, image_root):
        self.images = images
        self.captions = captions
        self.image_root = image_root

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        lesion_id = self.images[idx]
        caption = self.captions[idx]
        image_path = os.path.join(self.image_root, f"lesion_{lesion_id}.png")

        if not os.path.exists(image_path):
            print(f"File not found: {image_path}")
            return None, None

        try:
            image = preprocess(Image.open(image_path)).to(device)
            text = torch.tensor(tokenizer(caption)).to(device)
            return image, text
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return None, None

def collate_fn(batch):
    batch = [item for item in batch if item[0] is not None]
    if not batch:
        return None, None
    images, texts = zip(*batch)
    return torch.stack(images), torch.cat(texts)

# Paths
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--image_dir", type=str, required=True, help="Directory containing lesion PNG images")
args = parser.parse_args()
image_dir = args.image_dir


# Loaders
train_loader = DataLoader(LesionDataset(images, captions, image_dir), batch_size=64, shuffle=True, collate_fn=collate_fn)
eval_loader = DataLoader(LesionDataset(eval_images, eval_captions, image_dir), batch_size=64, shuffle=False, collate_fn=collate_fn)

# Optimizer and Loss
optimizer = Adam(model.parameters(), lr=5e-5)
criterion = CrossEntropyLoss()

# Train + Eval
for epoch in range(2):
    model.train()
    total_loss = 0
    for images, texts in train_loader:
        if images is None: continue
        optimizer.zero_grad()
        image_features, text_features, logit_scale = model(images, texts)
        logits = logit_scale * image_features @ text_features.T
        labels = torch.arange(len(images)).to(device)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} Train Loss: {total_loss / len(train_loader):.4f}")
    torch.save(model.state_dict(), f"epoch_{epoch+1}_biomedclip.pth")

    # Evaluation
    model.eval()
    total, correct, eval_loss = 0, 0, 0
    with torch.no_grad():
        for images, texts in eval_loader:
            if images is None: continue
            image_features, text_features, logit_scale = model(images, texts)
            logits = logit_scale * image_features @ text_features.T
            labels = torch.arange(len(images)).to(device)
            loss = criterion(logits, labels)
            eval_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = 100 * correct / total if total > 0 else 0
    print(f"Epoch {epoch+1} Eval Loss: {eval_loss / len(eval_loader):.4f}, Accuracy: {acc:.2f}%")
