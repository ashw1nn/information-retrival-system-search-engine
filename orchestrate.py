import subprocess
import itertools
from itertools import combinations
import os
import argparse
import csv
import json
import numpy as np
from scipy.stats import ttest_rel, wilcoxon

MODELS = ["tfidf", "lsa", "bm25"]
FLAG_COMBINATIONS = list(itertools.product([False, True], repeat=3))  # (rerank, expand, spell)

SCRIPT = "main.py"
BASE_COMMAND = ["python", SCRIPT, "-dataset", "cranfield/", "-out_folder", "output/"]

RESULT_DIR = "output/results"
EXPECTED_RUNS = len(MODELS) * len(FLAG_COMBINATIONS)

# Run all configurations
def run_all_configs():
    completed = 0
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    for model in MODELS:
        for flags in FLAG_COMBINATIONS:
            rerank, expand, spell = flags
            config_name = f"{model}_"
            if rerank: config_name += "rerank_"
            if expand: config_name += "expand_"
            if spell: config_name += "spell"
            config_name = config_name.rstrip('_') or "plain"
            expected_file = os.path.join(RESULT_DIR, f"{model}_{config_name}_scores_k10.json")

            if not args.force and os.path.exists(expected_file):
                print(f"✅ Skipping {expected_file} (already exists)")
                completed += 1
                continue

            cmd = BASE_COMMAND + ["-model", model]
            if not rerank: cmd.append("--no-rerank")
            if not expand: cmd.append("--no-expand")
            if not spell: cmd.append("--no-spell")

            print(f"\n🚀 Running config: {model}_{config_name}")
            subprocess.run(cmd)
            completed += 1

    return completed


def run_hypothesis_eval():
    print("\n📊 Running hypothesis evaluations...")
    results = []

    all_files = [f for f in os.listdir(RESULT_DIR) if f.endswith("scores_k10.json")]

    # Define your baseline
    baseline_file = "tfidf_plain_scores_k10.json"
    if baseline_file not in all_files:
        raise FileNotFoundError(f"Baseline file '{baseline_file}' not found in {RESULT_DIR}")

    base_path = os.path.join(RESULT_DIR, baseline_file)
    map_A, ndcg_A, prec_A, recall_A, fscore_A = load_scores(base_path)
    label_A = baseline_file.replace("_scores_k10.json", "")

    # Compare all other files against the baseline
    for comp in all_files:
        if comp == baseline_file:
            continue

        comp_path = os.path.join(RESULT_DIR, comp)
        map_B, ndcg_B, prec_B, recall_B, fscore_B = load_scores(comp_path)
        label_B = comp.replace("_scores_k10.json", "")

        csv_out = {
            "Comparison": f"{label_A} vs {label_B}",
            "MAP@10_mean_comp": round(np.mean(map_B), 4),
            "nDCG@10_mean_comp": round(np.mean(ndcg_B), 4),
            "Precision@10_mean_comp": round(np.mean(prec_B), 4),
            "Recall@10_mean_comp": round(np.mean(recall_B), 4),
            "FScore@10_mean_comp": round(np.mean(fscore_B), 4),
        }

        for metric_name, scores_X, scores_Y in [
            ("MAP@10", map_A, map_B),
            ("nDCG@10", ndcg_A, ndcg_B),
            ("Precision@10", prec_A, prec_B),
            ("Recall@10", recall_A, recall_B),
            ("FScore@10", fscore_A, fscore_B),
        ]:
            t_stat, t_p = ttest_rel(scores_X, scores_Y)
            diffs = scores_X - scores_Y
            if np.all(diffs == 0):
                w_p = "SKIPPED"
            else:
                _, w_p = wilcoxon(scores_X, scores_Y)

            csv_out[f"{metric_name} t_p"] = round(t_p, 4)
            csv_out[f"{metric_name} w_p"] = w_p if w_p == "SKIPPED" else round(w_p, 4)

        results.append(csv_out)

        if results:
            os.makedirs("output/hypothesis_results", exist_ok=True)
            csv_path = "output/hypothesis_results/summary.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)




def load_scores(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return (
        np.array(data['map']),
        np.array(data['ndcg']),
        np.array(data['precision']),
        np.array(data['recall']),
        np.array(data['fscore'])
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true', help="Force rerun all configs even if outputs exist")
    args = parser.parse_args()

    files_found = os.path.exists(RESULT_DIR) and len([f for f in os.listdir(RESULT_DIR) if f.endswith("scores_k10.json")]) >= EXPECTED_RUNS
    if files_found and not args.force:
        print("\n📁 All score files already exist. Skipping main.py runs.")
    else:
        run_all_configs()

    run_hypothesis_eval()