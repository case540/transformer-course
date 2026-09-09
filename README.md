# Transformers from First Principles

Three bite-sized, exercise-driven notebooks for learning Python, general machine
learning, neural networks, and transformers with PyTorch.

## Lessons

1. `notebooks/01_shakespeare_causal_transformer.ipynb` — character tokenization,
   next-token prediction, causal self-attention, a manual training loop, evaluation,
   TensorBoard, checkpointing, and text generation.
2. `notebooks/02_bidirectional_transformer_regression.ipynb` — synthetic sequence
   regression, feature scaling, train/validation/test splits, padding masks,
   bidirectional self-attention, regression metrics, baselines, and residual analysis.
3. `notebooks/03_transformer_from_components.ipynb` — a visual, intentionally
   simple look inside attention using three words, 2D embeddings, `Q = K = V`,
   hand-worked dot products, attention percentages, causal masking, a tiny block,
   and an optional visualization of real pretrained GloVe embeddings.

Read [`LESSON_PLAN.md`](LESSON_PLAN.md) for the teaching sequence and learning
objectives.

## Run locally on a Mac

Python 3.10+ is recommended. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
jupyter lab
```

Open a notebook and use **Kernel → Restart Kernel and Run All Cells**. PyTorch
will select Apple Silicon MPS when available, otherwise CPU. The default model is
deliberately small; use `FAST_MODE = True` while learning/debugging and switch it
to `False` for a longer run.

TensorBoard logs are written beneath `runs/`. In a notebook, run the included
`%tensorboard --logdir ...` cell, or from a terminal:

```bash
tensorboard --logdir runs
```

## Run in Google Colab

Upload or open any `.ipynb` file in Colab, then choose **Runtime → Run all**.
PyTorch is preinstalled. For faster training choose **Runtime → Change runtime
type → T4 GPU**. Files in a Colab session are temporary; download checkpoints or
mount Google Drive if you want to keep them.

## How to use the exercises

Work top-to-bottom. Each exercise asks for one small implementation and is followed
by a check cell. A check explains what is missing instead of crashing if the TODO
is still unfinished. The answer key at the bottom is independent and runnable, so
you can compare approaches or execute the complete pipeline.

## Validate the project

```bash
python scripts/validate_notebooks.py
```

The validator checks notebook JSON, required sections, answer keys, and code-cell
syntax without doing a full training run.
