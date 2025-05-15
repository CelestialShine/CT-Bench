
import json
from torch.utils.data import Dataset
import os

class TaskDataset(Dataset):
    def __init__(self, json_path, image_base_path='.'):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.image_base_path = image_base_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        q_type = item['q_type']
        base = {
            "qa_id": item.get("qa_id"),
            "q_type": q_type,
            "lesion_idx": item.get("lesion_idx"),
            "split": item.get("split"),
            "bbox_type": item.get("bbox_type"),
            "correct_choice": item.get("correct_choice")
        }

        correct_index = ord(item['correct_choice']) - ord('A')

        if q_type in ["img2txt", "img2attrib", "img2size"]:
            base.update({
                "image_paths": [os.path.join(self.image_base_path, item['image_path'])],
                "text_inputs": item['text'],
                "correct_index": correct_index
            })

        elif q_type in ["ct2txt", "ct2attrib"]:
            base.update({
                "image_paths": [os.path.join(self.image_base_path, p) for p in item['image_paths']],
                "text_inputs": item['text'],
                "correct_index": correct_index
            })

        elif q_type in ["txt2img", "txt2bbox"]:
            base.update({
                "image_paths": [os.path.join(self.image_base_path, p) for p in item['image_paths']],
                "text_inputs": [item['text']],
                "correct_index": correct_index
            })

        return base
