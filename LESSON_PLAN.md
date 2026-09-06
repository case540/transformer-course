# Lesson plan

## Audience and teaching approach

This series assumes no machine-learning experience and only beginning Python.
Readers familiar with R get short translations for indexing, dictionaries,
comprehensions, classes, tensor shapes, broadcasting, binding/stacking, tuple
unpacking, iterators, random state, device movement, and mutation. These “R bridge”
notes appear beside the Python that motivates them instead of in a separate glossary.
Every lesson uses this loop:

1. Explain the concept and why it exists.
2. Show the relevant PyTorch API and tensor shapes.
3. Ask for a small implementation.
4. Run an immediate, specific check.
5. Connect the result to the complete answer key.

The notebooks favor an explicit training loop over a high-level trainer because
the loop exposes gradients, optimizer steps, train/eval modes, data leakage, and
metrics. High-level alternatives are named after the mechanics are understood.

## Notebook 1 — causal text transformer

**Question:** Can a small model learn enough character-level structure from Tiny
Shakespeare to lower validation loss and generate play-like text?

Learning sequence:

- reproducibility, device choice, and Python/tensor notation;
- downloading and inspecting a 1.1 MB text corpus;
- characters versus tokens, vocabulary, integer encoding, and decoding;
- chronological train/validation/test splits and leakage;
- context windows, shifted targets, batches, and tensor shapes;
- embeddings and learned positional embeddings;
- queries, keys, values, scaled dot-product attention, multiple heads;
- causal masks and a concrete “future-token leak” demonstration;
- residual connections, layer normalization, feed-forward networks, dropout;
- logits, softmax, cross-entropy, perplexity, and random-guess baselines;
- autograd, AdamW, gradient clipping, training/evaluation modes;
- TensorBoard, checkpoints, sampling temperature, and top-k generation;
- underfitting/overfitting and controlled experiments.

Expected outcome: even a fast run should trend below the random-character loss.
A longer Mac/Colab run should produce recognizable punctuation, line breaks, and
speaker-like formatting, though not coherent Shakespeare.

## Notebook 2 — bidirectional transformer regression

**Question:** Can an encoder use the entire sequence to estimate a continuous
target, and what changes when there is no causal mask?

The synthetic dataset is intentional: it is fast, downloadable-data-free, and has
a known signal. Each variable-length sequence contains noisy measurements; the
target depends on global mean, trend, and a nonlinear interaction. This makes
attention useful and lets us test whether the model recovers known structure.

Learning sequence:

- regression versus classification/language modeling;
- reproducible synthetic data, visualization, and data-generating processes;
- train/validation/test purpose and fitting preprocessing on train only;
- standardization, baseline predictors, and leakage prevention;
- variable lengths, zero-padding, padding masks, and masked pooling;
- projecting numeric features into `d_model` dimensions;
- bidirectional self-attention and comparison with causal attention;
- sequence-to-one pooling and a linear regression head;
- MSE, MAE, RMSE, R², and why several metrics matter;
- minibatch training, early stopping, checkpoints, and TensorBoard;
- prediction/residual plots, subgroup errors, and iteration ideas.

Expected outcome: the transformer should beat the “predict the training mean”
baseline and show downward validation loss. The attention-mask experiment should
show that changing a future timestep can influence an earlier representation only
in the bidirectional model.

## Suggested schedule

- Session 1 (60–90 min): tensors, data inspection, tokenization, splitting.
- Session 2 (90 min): attention, masks, and architecture.
- Session 3 (60–90 min): training, metrics, TensorBoard, generation.
- Session 4 (60 min): regression data, preprocessing, and baselines.
- Session 5 (90 min): encoder regression, evaluation, and experiments.

Do not rush to maximize quality. Change one hyperparameter at a time, record the
result, and explain the observation before running another experiment.
