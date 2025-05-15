
from task_dataset_wrapper import TaskDataset
from task_dataset_evaluator import run_evaluation
import json

# === User Configuration ===
dataset_path = "qa_clip_all_final_corrected.json"  # Your dataset file
image_base_path = "/path/to/image/root"            # Change this to where your images are located
output_file = "biomedclip_eval_results.json"       # Output JSON for results

# === Load Dataset ===
dataset = TaskDataset(json_path=dataset_path, image_base_path=image_base_path)

# === Run Evaluation ===
results, stats = run_evaluation(dataset)

# === Save Results ===
with open(output_file, "w") as f:
    json.dump({"results": results, "task_stats": stats}, f, indent=2)

# === Print Summary ===
print("\nEvaluation Complete:")
for task, s in stats.items():
    acc = s['correct'] / s['total'] if s['total'] > 0 else 0
    print(f"  {task}: {acc:.4f} ({s['correct']}/{s['total']})")
