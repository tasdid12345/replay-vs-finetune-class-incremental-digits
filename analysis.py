"""Class-incremental digits: naive sequential MLP (baseline) vs. the same MLP with a small replay buffer (improved)."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

SEED = 42
SEEDS = [42, 43, 44]                              # both MLPs are randomized -> run each over these seeds
EPOCHS = 30                                       # passes over each task's (plus replay) data
PER_CLASS = 30                                    # replay exemplars kept per previously seen class
TASKS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]  # 5 class-incremental tasks, 2 digit classes each
SOURCE = "https://scikit-learn.org/stable/datasets/toy_dataset.html"


def run_sequence(X_tr, y_tr, X_te, y_te, seed, replay):
    """Learn tasks in order; task-agnostic prediction over all 10 outputs (no task ID)."""
    rng = np.random.default_rng(seed)
    clf = MLPClassifier(hidden_layer_sizes=(128,), random_state=seed)
    buf_X, buf_y = [], []                         # replay buffer (empty for the baseline)
    T = len(TASKS)
    # acc_mat[i, j]: accuracy on task j's TRAINING samples right after learning task i
    # (training split only, so the test set is touched once, at the end)
    acc_mat = np.zeros((T, T))
    for i, task in enumerate(TASKS):
        m = np.isin(y_tr, task)
        Xt, yt = X_tr[m], y_tr[m]
        if buf_X:                                 # replay: mix stored old-class exemplars into the new task
            Xt = np.vstack([Xt] + buf_X)
            yt = np.concatenate([yt] + buf_y)
        for _ in range(EPOCHS):                   # partial_fit = one shuffled epoch
            clf.partial_fit(Xt, yt, classes=np.arange(10))
        for j, tj in enumerate(TASKS):
            mj = np.isin(y_tr, tj)
            acc_mat[i, j] = np.mean(clf.predict(X_tr[mj]) == y_tr[mj])
        if replay:                                # keep PER_CLASS random exemplars of each new class
            for c in task:
                idx = rng.choice(np.where(y_tr == c)[0], PER_CLASS, replace=False)
                buf_X.append(X_tr[idx])
                buf_y.append(y_tr[idx])
    pred = clf.predict(X_te)                      # single final test evaluation
    # average forgetting over the first T-1 tasks: just-after accuracy minus final accuracy
    forgetting = np.mean([acc_mat[j, j] - acc_mat[-1, j] for j in range(T - 1)])
    first = np.isin(y_te, TASKS[0])
    return {
        "acc": float(np.mean(pred == y_te)),
        "f1": float(f1_score(y_te, pred, average="macro")),
        "first_task_acc": float(np.mean(pred[first] == y_te[first])),
        "forgetting": float(forgetting),
    }


def main():
    digits = load_digits()                        # bundled with scikit-learn, no download
    X = digits.data / 16.0                        # fixed rescaling of pixels 0..16 to [0,1]
    y = digits.target
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    print(f"rows={len(y)} train={len(y_tr)} test={len(y_te)}")

    res = {"baseline": [], "improved": []}
    for s in SEEDS:
        print(f"seed {s}: baseline (sequential fine-tuning)")
        res["baseline"].append(run_sequence(X_tr, y_tr, X_te, y_te, s, replay=False))
        print(f"seed {s}: improved (replay {PER_CLASS}/class)")
        res["improved"].append(run_sequence(X_tr, y_tr, X_te, y_te, s, replay=True))

    def agg(name, key):
        v = np.array([r[key] for r in res[name]])
        return float(v.mean()), float(v.std())

    b_acc, b_sd = agg("baseline", "acc")
    i_acc, i_sd = agg("improved", "acc")
    metrics = {
        "baseline_score": b_acc, "baseline_score_std": b_sd,
        "improved_score": i_acc, "improved_score_std": i_sd,
        "improvement": i_acc - b_acc,             # higher is better -> improved - baseline
        "baseline_f1": agg("baseline", "f1")[0], "improved_f1": agg("improved", "f1")[0],
        "baseline_first_task_acc": agg("baseline", "first_task_acc")[0],
        "improved_first_task_acc": agg("improved", "first_task_acc")[0],
        "baseline_forgetting": agg("baseline", "forgetting")[0],
        "improved_forgetting": agg("improved", "forgetting")[0],
        "n_seeds": len(SEEDS),
    }
    if not all(np.isfinite(v) for v in metrics.values()):
        print("non-finite metric, cannot produce meaningful results")
        sys.exit(1)

    results = {
        "question": "Does a small replay buffer of old-class exemplars let a task-agnostic MLP keep earlier "
                    "classes better than naive sequential fine-tuning on a class-incremental digits benchmark?",
        "method": "5 class-incremental tasks (2 digit classes each), task-agnostic evaluation over 10 outputs; "
                  "80/20 stratified split; baseline MLP fine-tuned sequentially vs. same MLP with 30 replayed "
                  "exemplars per old class; seeds 42-44.",
        "dataset": {
            "name": "scikit-learn digits (load_digits)",
            "source_url": SOURCE,
            "n_rows": int(len(y)),
            "n_train": int(len(y_tr)),
            "n_test": int(len(y_te)),
            "same_as_paper": False,
            "baseline_is_papers_method": False,
        },
        "metric_name": "final average accuracy over all 10 classes on held-out test split",
        "higher_is_better": True,
        "baseline": "MLPClassifier (1 hidden layer, 128 units), sequentially fine-tuned on each task's data only",
        "improved": f"Same MLP plus a replay buffer of {PER_CLASS} exemplars per previously seen class, "
                    "mixed into each new task's training data",
        "metrics": metrics,
        "notes": "Naive stand-in for the paper's hierarchical model, not its method. Hyper-parameters fixed a priori "
                 "(no tuning, so no CV selection was needed). Forgetting and per-task accuracies are computed on the "
                 "training split to keep the test set single-use. The improved model sees more samples per task "
                 "(replay) with the same epoch count, so compute is not perfectly matched. Std is over 3 seeds "
                 "(population std). Single small dataset; results are indicative only.",
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    os.makedirs("figures", exist_ok=True)         # one bar chart: final accuracy with seed std
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.bar(["baseline", "improved"], [b_acc, i_acc], yerr=[b_sd, i_sd], capsize=6, color=["gray", "tab:blue"])
    ax.set_ylabel("final test accuracy (10 classes)")
    ax.set_title("Class-incremental digits")
    fig.tight_layout()
    fig.savefig("figures/final_accuracy.png", dpi=120)
    print(f"baseline={b_acc:.4f}±{b_sd:.4f}  improved={i_acc:.4f}±{i_sd:.4f}  results.json written")


if __name__ == "__main__":
    main()
