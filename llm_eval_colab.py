# ✅ LLM Evaluation Notebook (Sequential Inference, CUDA, Summary File)

# 1. Setup
!pip install -q transformers accelerate bitsandbytes datasets rouge-score evaluate

import os
import json
import torch
import evaluate
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# 2. Load local JSONL validation set
def load_local_dataset(filepath):
    with open(filepath, 'r') as f:
        data = [json.loads(line) for line in f]
    return data

# 3. Evaluate predictions
def evaluate_predictions(preds, refs):
    exact_match = sum(p.strip() == r.strip() for p, r in zip(preds, refs)) / len(refs)
    rouge = evaluate.load("rouge")
    rouge_score = rouge.compute(predictions=preds, references=refs, use_stemmer=True)
    return {
        "accuracy": round(exact_match * 100, 2),
        "rougeL": round(rouge_score["rougeL"] * 100, 2)
    }

# 4. Run model inference using raw generate (faster than pipeline)
def run_model_inference(model_name_or_path, test_data):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",
        load_in_8bit=True
    )
    model.eval()

    predictions = []
    for item in tqdm(test_data):
        prompt = item["prompt"]
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False
            )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predictions.append(generated_text)

    del model
    torch.cuda.empty_cache()

    return predictions

# 5. Save and evaluate a single model
def run_single_model_evaluation(model_id, model_path, test_data, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    refs = [item["expected_output"] for item in test_data]

    print(f"\n🚀 Running inference for: {model_id}")
    preds = run_model_inference(model_path, test_data)

    # Evaluate
    scores = evaluate_predictions(preds, refs)

    # Save outputs
    output_file = f"{output_dir}/{model_id}_outputs.jsonl"
    with open(output_file, "w") as f:
        for item, pred in zip(test_data, preds):
            f.write(json.dumps({
                "prompt": item["prompt"],
                "expected_output": item["expected_output"],
                "model_output": pred
            }) + "\n")

    # Save evaluation scores
    score_file = f"{output_dir}/{model_id}_scores.json"
    with open(score_file, "w") as f:
        json.dump(scores, f, indent=2)

    print(f"✅ {model_id} Evaluation Complete: Accuracy = {scores['accuracy']}%, ROUGE-L = {scores['rougeL']}%")

# 6. Display summary from all score files
def display_final_summary(output_dir):
    print("\n📊 Final Evaluation Summary:\n")
    for file in os.listdir(output_dir):
        if file.endswith("_scores.json"):
            model_id = file.replace("_scores.json", "")
            with open(os.path.join(output_dir, file), 'r') as f:
                scores = json.load(f)
                print(f"{model_id}: Accuracy = {scores['accuracy']}%, ROUGE-L = {scores['rougeL']}%")

# 7. Example usage for individual model run:
# test_data = load_local_dataset("/content/validation.jsonl")
# run_single_model_evaluation("hermes", "teknium/OpenHermes-2.5-Mistral-7B", test_data, "outputs")

# Later, after all models are run:
# display_final_summary("outputs")
