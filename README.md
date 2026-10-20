# Replay buffers for task-agnostic class-incremental digits

## Motivation
Continual learning systems must learn new classes without forgetting old ones. The paper "Hierarchically structured task-agnostic continual learning" (2023) studies task-agnostic continual learning, where no task identity is available at test time. This experiment does not reproduce that paper's hierarchical model; it asks a simpler question about a naive baseline and a common remedy.

## Question
Does a small replay buffer of old-class exemplars let a task-agnostic MLP keep earlier classes better than naive sequential fine-tuning on a class-incremental digits benchmark?

## Dataset
The scikit-learn built-in digits dataset (`load_digits`, 1,797 images, 10 classes), loaded locally with no download. It is not the paper's own dataset. The data is split once into 80% train and 20% test, stratified, with seed 42, and the training part is divided into five tasks of two classes each.

## Baseline vs. improved model
- **Baseline:** a standard `MLPClassifier` with one hidden layer of 128 units, fine-tuned sequentially on each task's data only. It is a naive stand-in, not the paper's method.
- **Improved:** the same MLP, but it keeps 30 exemplars per previously seen class in a replay buffer and mixes them into each new task's training.

Both models are evaluated task-agnostically over all 10 outputs.

## How to run
```
pip install -r requirements.txt
python analysis.py
```

## Results
Results are written to `results.json` (and a bar chart to `figures/`) when the script runs.

## Results (latest run)

On the scikit-learn digits benchmark (final average accuracy over 10 classes on the same 360-sample held-out test split, 3 seeds), a standard sequentially fine-tuned MLP reached 19%, while the same MLP with 30 replayed exemplars per old class reached 87% (±1.5%).

| Metric | Value |
|---|---|
| baseline_score | 0.1861 |
| baseline_score_std | 0 |
| improved_score | 0.8685 |
| improved_score_std | 0.01544 |
| improvement | 0.6824 |
| baseline_f1 | 0.06514 |
| improved_f1 | 0.873 |
| baseline_first_task_acc | 0 |
| improved_first_task_acc | 0.7917 |
| baseline_forgetting | 0.9919 |
| improved_forgetting | 0.1537 |
| n_seeds | 3 |

Run on 2026-10-20. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Naive stand-in for the paper's hierarchical model, not its method. Hyper-parameters fixed a priori (no tuning, so no CV selection was needed). Forgetting and per-task accuracies are computed on the training split to keep the test set single-use. The improved model sees more samples per task (replay) with the same epoch count, so compute is not perfectly matched. Std is over 3 seeds (population std). Single small dataset; results are indicative only.

## Reference

This project is motivated by the following paper:

> H Hihn, DA Braun (2023). *Hierarchically structured task-agnostic continual learning*. Machine Learning. https://link.springer.com/article/10.1007/s10994-022-06283-9

Paper author: Prof. Daniel Alexander Braun, Ulm University (daniel.braun@uni-ulm.de)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
