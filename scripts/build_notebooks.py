"""Generate the course notebooks from reviewable Python source."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(text).strip() + "\n"}


def code(text: str, tags: list[str] | None = None) -> dict:
    metadata = {"tags": tags} if tags else {}
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": metadata,
        "outputs": [],
        "source": dedent(text).strip() + "\n",
    }


def exercise(title: str, prompt: str, starter: str, check: str) -> list[dict]:
    return [
        md(f"""### Exercise — {title}

{prompt}

Replace the `None`/`TODO` portion below. The next cell is a small unit test: green
output means the behavior and important shapes are correct, not that there is only
one valid solution."""),
        code(starter, ["exercise"]),
        code(check, ["check"]),
    ]


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "colab": {"name": "Transformers lesson", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


TEXT_CELLS = [
    md("""# 1 — Build a causal text transformer from scratch

**Learning goal:** train a small decoder-only transformer to predict the next
character in Tiny Shakespeare, understand every important tensor, and generate a
sample. This is a learning model, not a production LLM.

We use PyTorch's low-level building blocks and write the training loop ourselves.
The answer key at the bottom is a complete runnable path. Start with `FAST_MODE =
True`; correctness and a downward trend matter more than literary output.

> A *model* is a parameterized function. *Training* adjusts its parameters to make
> its predictions less wrong on examples. A transformer is a neural-network family
> whose key operation, attention, lets each sequence position combine information
> from other positions."""),
    md("""## Map of the pipeline

`text → tokens → context/target batches → embeddings → transformer blocks → logits
→ cross-entropy loss → gradients → AdamW update → validation → generation`

Three splits have different jobs:

- **train:** gradients update model parameters;
- **validation:** choose hyperparameters and detect overfitting;
- **test:** one final, relatively unbiased report—do not repeatedly tune on it.

We split contiguous text chronologically. Randomly splitting individual overlapping
windows can leak almost-identical text across splits."""),
    code("""# Imports, reproducibility, and hardware selection
from pathlib import Path
from urllib.request import urlretrieve
import math, random, time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
FAST_MODE = True
print("PyTorch:", torch.__version__, "| device:", device, "| fast mode:", FAST_MODE)
"""),
    md("""### Python bridge for R users

- Python indexing starts at 0. `x[:10]` means the first ten values; the upper bound
  is excluded. It resembles R's `head(x, 10)`, but not R's 1-based indexes.
- A dictionary such as `{"a": 0}` is a named lookup table, similar to a named R
  vector/list. `d[k]` looks up key `k`.
- `[f(x) for x in items]` is a list comprehension, much like `lapply(items, f)`.
- Tensor shapes are written `(B, T, C)`: batch, time/sequence, channels/features.
- A Python `class` bundles parameters and behavior. PyTorch models subclass
  `nn.Module`; `forward` defines the computation.
- `@torch.no_grad()` is a decorator: it changes how the function below runs, here
  disabling gradient bookkeeping during evaluation."""),
    md("""## 1. Data and tokenization

Tiny Shakespeare is about 1.1 MB of text. A **token** is one discrete unit presented
to the model. Modern LLMs usually use subword tokens; here each distinct character
is a token. Character tokenization is transparent and needs only ~65 vocabulary
entries, but sequences are longer and semantic units such as words are not explicit.

Tokenization maps text to integer IDs. IDs are labels, not magnitudes: token 40 is
not “twice” token 20. An embedding table later learns a vector for every ID.

Dataset: [Karpathy's Tiny Shakespeare](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt)  
API: [`torch.tensor`](https://docs.pytorch.org/docs/stable/generated/torch.tensor.html)"""),
    code("""DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
data_dir = Path("data"); data_dir.mkdir(exist_ok=True)
data_path = data_dir / "tiny_shakespeare.txt"
if not data_path.exists():
    urlretrieve(DATA_URL, data_path)
text = data_path.read_text(encoding="utf-8")
print(f"{len(text):,} characters | preview:\\n{text[:300]}")
"""),
]

TEXT_CELLS += exercise(
    "build a character vocabulary",
    "Create a sorted list of unique characters and two dictionaries: character to integer and integer to character. Sorting makes IDs reproducible.",
    """chars = None  # TODO: sorted(set(text))
stoi = None   # TODO: {character: integer, ...}
itos = None   # TODO: {integer: character, ...}""",
    """if chars is None:
    print("🟡 Not attempted yet. Hint: enumerate(chars) yields (index, value).")
else:
    assert len(chars) == len(set(text)) and chars == sorted(chars)
    assert all(itos[stoi[ch]] == ch for ch in chars)
    print(f"🟢 Vocabulary is reversible and has {len(chars)} tokens.")""",
)

TEXT_CELLS += [
    md("""## 2. Encode, decode, and split

Encoding is a deterministic data transformation, not something learned. We turn
the entire corpus into a one-dimensional `torch.long` tensor because embedding
layers require integer indexes. Decoding reverses that operation for inspection.

The 80/10/10 split is a convention, not a law. The key rule is that test data stays
untouched until the end. For time-ordered or authored data, preserve order unless
you have a reason not to."""),
]
TEXT_CELLS += exercise(
    "write encode and decode",
    "Use the vocabulary dictionaries. A function is introduced by `def`; its indented body ends when indentation ends.",
    """def encode(s):
    return None  # TODO: a list of IDs

def decode(ids):
    return None  # TODO: one string""",
    """if encode("Hi") is None:
    print("🟡 Not attempted yet. Hint: ''.join(...) combines characters.")
else:
    probe = text[:100]
    assert decode(encode(probe)) == probe
    print("🟢 Round trip passed:", repr(decode(encode("ROMEO"))))""",
)
TEXT_CELLS += exercise(
    "make train/validation/test tensors",
    "Encode the text once, then take contiguous 80%, 10%, and 10% slices. Use dtype `torch.long`.",
    """data = None
train_data = None
val_data = None
test_data = None""",
    """if data is None:
    print("🟡 Not attempted. Hint: n = len(data); data[:int(.8*n)] is train.")
else:
    assert data.dtype == torch.long
    assert len(train_data)+len(val_data)+len(test_data) == len(data)
    assert torch.equal(torch.cat([train_data,val_data,test_data]), data)
    print("🟢 Split sizes:", *(len(x) for x in (train_data,val_data,test_data)))""",
)
TEXT_CELLS += [
    md("""## 3. Context windows and shifted labels

Language modeling uses self-supervision: the raw text supplies its own labels. If a
window is `Hell`, its targets are `ello`. At every position, the model predicts the
next token. A context length `T` yields `T` supervised predictions per example.

`batch_size` (`B`) is the number of windows processed before one optimizer update.
`block_size` (`T`) is the maximum context length. Larger values cost more memory;
self-attention work grows roughly with `T²`.

We randomly sample windows *inside one already-separated split*. This is stochastic
gradient descent: each update sees a small estimate of the full-data gradient."""),
    code("""# Temporary reference encoding so later exercises can run independently.
_chars = sorted(set(text)); _stoi = {ch:i for i,ch in enumerate(_chars)}
_data = torch.tensor([_stoi[ch] for ch in text], dtype=torch.long)
_n = len(_data); _train = _data[:int(.8*_n)]; _val = _data[int(.8*_n):int(.9*_n)]
batch_size = 16 if FAST_MODE else 32
block_size = 64 if FAST_MODE else 128
"""),
]
TEXT_CELLS += exercise(
    "sample a shifted minibatch",
    "Implement `get_batch(source)`. Sample `batch_size` legal start indexes, stack input windows and their one-character-shifted targets, then move both to `device`.",
    """def get_batch(source):
    # TODO
    return None, None""",
    """xb, yb = get_batch(_train)
if xb is None:
    print("🟡 Not attempted. Useful APIs: torch.randint, torch.stack.")
else:
    assert xb.shape == yb.shape == (batch_size, block_size)
    assert torch.equal(xb[:,1:], yb[:,:-1])
    assert xb.device == device
    print("🟢 Batch shape:", tuple(xb.shape), "| first pair:", xb[0,:5].tolist(), yb[0,:5].tolist())""",
)
TEXT_CELLS += [
    md("""## 4. What attention computes

For every token representation `x`, learned linear layers create a **query** (what
this position seeks), **key** (what this position advertises), and **value** (the
information it can contribute). Similarity scores are `Q @ Kᵀ / sqrt(head_dim)`.
Softmax turns scores into nonnegative weights summing to one, then weights mix `V`.

Multiple heads repeat this with different learned projections, allowing different
relationships. The outputs are concatenated and projected. Positional embeddings
are essential because plain attention has no inherent word order.

For next-token prediction, position `t` must not inspect tokens after `t`; otherwise
training leaks the answer. A **causal mask** sets future attention scores to negative
infinity before softmax. Regression in Notebook 2 intentionally omits this mask.

Reference: [Attention Is All You Need](https://arxiv.org/abs/1706.03762),
[`nn.MultiheadAttention`](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)"""),
    code("""# Visualize a causal mask: 0 = visible, -inf = forbidden.
demo_mask = torch.triu(torch.full((8, 8), float("-inf")), diagonal=1)
plt.figure(figsize=(4, 3)); plt.imshow(torch.isfinite(demo_mask), cmap="Blues")
plt.xlabel("key position"); plt.ylabel("query position"); plt.title("Causal visibility")
plt.colorbar(label="can attend"); plt.show()
"""),
    md("""## 5. Architecture and parameters

- `vocab_size`: number of possible character classes.
- `d_model`: width of each token representation; larger means more capacity/cost.
- `n_head`: parallel attention heads; `d_model` must divide evenly by this.
- `n_layer`: transformer blocks stacked in depth.
- `dim_feedforward`: hidden width of each block's position-wise MLP.
- `dropout`: randomly zeros activations only during training to reduce overfitting.
- **Residual connections** add a block's input to its output, aiding gradient flow.
- **Layer normalization** stabilizes each token's feature scale.
- The final linear head maps `d_model` features to one logit per vocabulary token.

A **logit** is an unrestricted score, not a probability. Cross-entropy internally
applies log-softmax and rewards a high score for the correct next token. Random
guess loss is `ln(vocab_size)` and perplexity is `exp(loss)`.

Reference: [`TransformerEncoderLayer`](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html),
[`CrossEntropyLoss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)"""),
]

TEXT_CELLS += exercise(
    "define the language model",
    "Complete the model using token + position embeddings, `TransformerEncoder` with `batch_first=True`, a causal mask, final layer norm, and vocabulary projection.",
    """class TinyCausalTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=96, n_head=4, n_layer=2, dropout=.1, max_len=128):
        super().__init__()
        # TODO: define layers

    def forward(self, idx, targets=None):
        # TODO: return logits shaped (B,T,V), and optional scalar loss
        return None, None""",
    """try:
    candidate = TinyCausalTransformer(len(_chars), max_len=block_size).to(device)
    logits, loss = candidate(xb, yb)
    assert logits.shape == (batch_size, block_size, len(_chars))
    assert loss.ndim == 0 and torch.isfinite(loss)
    print("🟢 Forward pass works; initial loss:", round(loss.item(), 3))
except Exception as exc:
    print("🟡 Finish the model, then rerun. Current issue:", type(exc).__name__, exc)""",
)

TEXT_CELLS += [
    md("""## 6. Training and evaluation

One optimizer step is: clear old gradients → forward pass → loss → backward pass →
optionally clip gradients → update parameters. `AdamW` adapts each parameter's step
size and decouples weight decay. The **learning rate** is often the most important
hyperparameter: too high diverges, too low crawls.

Evaluation uses `model.eval()` (disables dropout), `torch.no_grad()` (saves memory),
and multiple batches (less noisy). Never call `backward()` or `optimizer.step()` on
validation/test data.

Reference: [`AdamW`](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html),
[`clip_grad_norm_`](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html),
[TensorBoard](https://docs.pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html)"""),
]

TEXT_CELLS += exercise(
    "write one optimizer step",
    "Given `model`, `optimizer`, `xb`, and `yb`, write the five core lines. Clip total gradient norm to 1.0 before stepping.",
    """# optimizer.zero_grad(set_to_none=True)
# TODO: forward, backward, clip, step""",
    """print("Self-check questions: Did zero_grad come before backward? Did optimizer.step come last?")""",
)

TEXT_CELLS += [
    md("""## 7. Generation is repeated classification

Generation feeds the current context through the model, selects from the final
position's next-token distribution, appends that token, and repeats. **Temperature**
divides logits: below 1 is safer/sharper; above 1 is more random. **Top-k** sampling
keeps only the k highest-scoring choices. Greedy argmax can become repetitive.

Training and generation differ: training predicts all positions in parallel under
a causal mask; generation creates one new position at a time."""),
    md("""# Answer key — complete runnable implementation

Try the exercises first. This section deliberately uses names prefixed with `ans_`
so it does not depend on incomplete exercise cells. Running from here trains a fresh
small model, logs metrics, saves the best checkpoint, evaluates once on test data,
and generates a sample."""),
    code("""# Answer 1–3: vocabulary, encoding, and leakage-safe split
ans_chars = sorted(set(text))
ans_stoi = {ch: i for i, ch in enumerate(ans_chars)}
ans_itos = {i: ch for i, ch in enumerate(ans_chars)}
ans_encode = lambda s: [ans_stoi[ch] for ch in s]
ans_decode = lambda ids: "".join(ans_itos[int(i)] for i in ids)
ans_data = torch.tensor(ans_encode(text), dtype=torch.long)
n = len(ans_data)
ans_train = ans_data[:int(.8*n)]
ans_val = ans_data[int(.8*n):int(.9*n)]
ans_test = ans_data[int(.9*n):]
print(len(ans_chars), "tokens; random baseline loss", round(math.log(len(ans_chars)), 3))
"""),
    code("""# Answer 4: random contiguous batches
def ans_get_batch(source):
    starts = torch.randint(0, len(source) - block_size - 1, (batch_size,))
    x = torch.stack([source[i:i+block_size] for i in starts])
    y = torch.stack([source[i+1:i+block_size+1] for i in starts])
    return x.to(device), y.to(device)

ax, ay = ans_get_batch(ans_train)
assert ax.shape == ay.shape == (batch_size, block_size)
"""),
    code("""# Answer 5: causal transformer
class AnswerCausalTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=96, n_head=4, n_layer=2, dropout=.1, max_len=128):
        super().__init__()
        self.max_len = max_len
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_head, 4*d_model, dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, n_layer, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        if T > self.max_len: raise ValueError(f"T={T} exceeds max_len={self.max_len}")
        pos = torch.arange(T, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(pos)[None, :, :]
        mask = torch.triu(torch.full((T, T), float("-inf"), device=idx.device), diagonal=1)
        x = self.blocks(x, mask=mask)
        logits = self.lm_head(self.norm(x))
        loss = None if targets is None else F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, new_tokens, temperature=0.8, top_k=20):
        self.eval()
        for _ in range(new_tokens):
            logits, _ = self(idx[:, -self.max_len:])
            logits = logits[:, -1, :] / temperature
            if top_k:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, 1)), dim=1)
        return idx

ans_model = AnswerCausalTransformer(len(ans_chars), max_len=block_size).to(device)
print(f"Parameters: {sum(p.numel() for p in ans_model.parameters()):,}")
"""),
    code("""# Answer 6: evaluation and explicit training loop
@torch.no_grad()
def estimate_loss(model, source, batches=5):
    model.eval(); values = []
    for _ in range(batches):
        x, y = ans_get_batch(source); _, loss = model(x, y); values.append(loss.item())
    model.train()
    return float(np.mean(values))

steps = 80 if FAST_MODE else 1000
eval_every = 20 if FAST_MODE else 100
optimizer = torch.optim.AdamW(ans_model.parameters(), lr=3e-4, weight_decay=.01)
writer = SummaryWriter("runs/shakespeare")
Path("checkpoints").mkdir(exist_ok=True)
history = {"step": [], "train": [], "val": []}; best_val = float("inf")

for step in range(steps + 1):
    if step % eval_every == 0:
        tr = estimate_loss(ans_model, ans_train); va = estimate_loss(ans_model, ans_val)
        history["step"].append(step); history["train"].append(tr); history["val"].append(va)
        writer.add_scalars("loss", {"train": tr, "validation": va}, step)
        print(f"step {step:4d} | train {tr:.3f} | val {va:.3f} | ppl {math.exp(va):.1f}")
        if va < best_val:
            best_val = va; torch.save(ans_model.state_dict(), "checkpoints/shakespeare_best.pt")
    if step == steps: break
    xb, yb = ans_get_batch(ans_train)
    optimizer.zero_grad(set_to_none=True)
    _, loss = ans_model(xb, yb)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(ans_model.parameters(), 1.0)
    optimizer.step()
    writer.add_scalar("train/grad_norm", float(grad_norm), step)
writer.close()
"""),
    code("""# Learning curves and final held-out test evaluation
plt.plot(history["step"], history["train"], marker="o", label="train")
plt.plot(history["step"], history["val"], marker="o", label="validation")
plt.axhline(math.log(len(ans_chars)), color="gray", ls="--", label="random baseline")
plt.xlabel("optimizer step"); plt.ylabel("cross-entropy"); plt.legend(); plt.show()
ans_model.load_state_dict(torch.load("checkpoints/shakespeare_best.pt", map_location=device, weights_only=True))
test_loss = estimate_loss(ans_model, ans_test, batches=10)
print(f"Test loss {test_loss:.3f}; perplexity {math.exp(test_loss):.1f}")
"""),
    code("""# Generate. FAST_MODE learns formatting before language; longer training is more legible.
prompt = "ROMEO:\\n"
context = torch.tensor([ans_encode(prompt)], dtype=torch.long, device=device)
sample = ans_model.generate(context, new_tokens=300, temperature=.8, top_k=20)
print(ans_decode(sample[0].tolist()))
"""),
    code("""# Optional TensorBoard inside Jupyter/Colab
%load_ext tensorboard
%tensorboard --logdir runs/shakespeare
""", ["skip-validation"]),
    md("""## Interpret, debug, and iterate

Healthy signs: training and validation loss both fall; validation may be noisy. If
training falls while validation rises, the model is overfitting. If neither falls,
check shifted targets and masks first, then try learning rate/capacity/steps. NaNs
often indicate an excessive learning rate or unstable values.

Suggested controlled experiments (change one thing and record train/validation):

1. Remove positional embeddings. What ordering ability remains?
2. Temporarily remove the causal mask. Why can validation loss look deceptively good?
3. Compare greedy, temperature 0.5/1.2, and top-k sampling.
4. Double context length; note both quality and time/memory.
5. Compare character tokens with a subword tokenizer conceptually.

Production LLMs add enormous datasets/models, sophisticated tokenizers, distributed
training, schedules, mixed precision, and alignment—but the central next-token
objective and attention mechanism here are genuine."""),
    md("""## References and next reading

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- PyTorch [`nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html)
- PyTorch [`Embedding`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html)
- PyTorch [`TransformerEncoder`](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html)
- PyTorch [Autograd tutorial](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)
- Dataset source: [Tiny Shakespeare](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare)

**Completion check:** explain tokens, `(B,T,C)`, causal leakage, logits, loss,
backpropagation, validation, and temperature in your own words. If any explanation
is fuzzy, rerun its smallest visualization or shape check."""),
]


REG_CELLS = [
    md("""# 2 — Bidirectional transformers for regression

**Learning goal:** predict one continuous number from a variable-length numeric
sequence using a transformer encoder **without a causal mask**.

Unlike next-token generation, every observed timestep is available when estimating
the target. Earlier representations may therefore attend to later observations.
We will create a dataset with known structure, beat a simple baseline, inspect
errors, and directly test bidirectional information flow."""),
    md("""## Problem and pipeline

Imagine each row is a short sensor history with two features: a noisy signal and a
control variable. The target depends on the signal's global average, its trend, and
an interaction. Sequences have different lengths.

`generate → split → fit train-only scaler → Dataset/DataLoader → pad + mask →
numeric projection + positions → bidirectional encoder → masked mean → regression
head → MSE → optimize → MAE/RMSE/R² + residual plots`

This synthetic task is scientifically useful: the true data-generating process is
known, so a bug cannot hide behind mysterious real-world data."""),
    code("""from pathlib import Path
import math, random
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

SEED = 7
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): device = torch.device("cuda")
elif torch.backends.mps.is_available(): device = torch.device("mps")
else: device = torch.device("cpu")
FAST_MODE = True
print(torch.__version__, device)
"""),
    md("""### Python bridge

`range(n)` produces 0 through n-1. A tuple such as `(x, y)` groups values. `*batch`
unpacks a tuple into function arguments (similar in spirit to R's `do.call`). A
leading underscore in `_private` is only a convention. `None` is Python's missing
object value; unlike numeric `NaN`, it has no arithmetic meaning."""),
    md("""## 1. Generate and visualize data

Each example has length 12–40 and shape `(T, 2)`. The target is continuous, so this
is **regression**. Classification chooses categories; language modeling performs a
classification at every token position.

The target formula is never provided to the model. It must approximate the mapping
from examples. Added noise creates irreducible error: even a perfect model should
not achieve exactly zero test MSE."""),
    code("""def make_example(rng, min_len=12, max_len=40):
    length = int(rng.integers(min_len, max_len + 1))
    t = np.linspace(0, 1, length, dtype=np.float32)
    level = rng.normal(0, 1); slope = rng.normal(0, .8)
    control = rng.uniform(-1, 1, size=length).astype(np.float32)
    signal = (level + slope*t + .35*np.sin(2*np.pi*t) + rng.normal(0,.12,length)).astype(np.float32)
    x = np.stack([signal, control], axis=1)
    y = 1.8*signal.mean() + 1.2*(signal[-1]-signal[0]) + .7*(signal*control).mean() + rng.normal(0,.12)
    return torch.tensor(x), torch.tensor(y, dtype=torch.float32)

rng = np.random.default_rng(SEED)
examples = [make_example(rng) for _ in range(500 if FAST_MODE else 2000)]
print("examples:", len(examples), "first x/y shapes:", examples[0][0].shape, examples[0][1].shape)
for x, y in examples[:4]: plt.plot(x[:,0], alpha=.8, label=f"y={y:.2f}")
plt.xlabel("timestep"); plt.ylabel("signal"); plt.legend(); plt.show()
"""),
]

REG_CELLS += exercise(
    "split by example",
    "Use `random_split` for 70% train, 15% validation, and the remainder test. Pass a seeded `torch.Generator` so the split is reproducible.",
    """train_set = None
val_set = None
test_set = None""",
    """if train_set is None:
    print("🟡 Not attempted. API: random_split(examples, [n_train,n_val,n_test], generator=...).")
else:
    assert len(train_set)+len(val_set)+len(test_set) == len(examples)
    assert set(train_set.indices).isdisjoint(test_set.indices)
    print("🟢 Split sizes:", len(train_set), len(val_set), len(test_set))""",
)

REG_CELLS += [
    md("""## 2. Preprocessing without leakage

Features measured in different units can make optimization difficult. Standardize
each feature as `(x - train_mean) / train_std`. Fit these statistics on **training
data only**. Using validation/test statistics leaks information from the future
evaluation distribution into training.

Targets can also be standardized; this keeps early gradients in a convenient range.
Convert predictions back to original units before reporting metrics."""),
]

REG_CELLS += exercise(
    "fit train-only scaling statistics",
    "Concatenate timesteps from training examples, then compute feature-wise mean/std and scalar target mean/std. Use population standard deviation (`correction=0`).",
    """# Use the reference indices only if your split exercise is unfinished.
_gen = torch.Generator().manual_seed(SEED)
_train, _val, _test = random_split(examples, [350,75,75], generator=_gen)
feature_mean = None; feature_std = None; target_mean = None; target_std = None""",
    """if feature_mean is None:
    print("🟡 Not attempted. Hint: torch.cat([x for x,y in _train], dim=0).")
else:
    assert feature_mean.shape == feature_std.shape == (2,)
    assert torch.all(feature_std > 0) and target_std > 0
    print("🟢 Feature mean/std:", feature_mean, feature_std)""",
)

REG_CELLS += [
    md("""## 3. Padding, masks, and DataLoader

A minibatch tensor must be rectangular, but sequence lengths differ. We pad shorter
examples with zeros to the longest sequence in that batch and create a Boolean
**padding mask** shaped `(B,T)`. `True` means “ignore this padded position.”

This mask is not causal. It hides nonexistent padding but leaves all real past and
future positions mutually visible. Dynamic per-batch padding wastes less work than
padding every example to the dataset maximum.

`Dataset` defines how to retrieve one item; `DataLoader` batches, optionally
shuffles, and calls `collate_fn` to combine variable-size items.

Reference: [`DataLoader`](https://docs.pytorch.org/docs/stable/data.html),
[`pad_sequence`](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_sequence.html)"""),
]

REG_CELLS += exercise(
    "collate variable-length sequences",
    "Write a collate function returning padded `x`, targets `y`, and a Boolean padding mask. You may use `pad_sequence(..., batch_first=True)`.",
    """def collate_batch(batch):
    # batch is a list of (x, y) tuples
    return None, None, None""",
    """px, py, pmask = collate_batch(examples[:4])
if px is None:
    print("🟡 Not attempted. Build lengths, then compare arange(max_len) >= lengths[:,None].")
else:
    assert px.ndim == 3 and py.shape == (4,) and pmask.shape == px.shape[:2]
    assert pmask.dtype == torch.bool
    assert torch.all(px[pmask] == 0)
    print("🟢 Shapes x/y/mask:", px.shape, py.shape, pmask.shape)""",
)

REG_CELLS += [
    md("""## 4. Encoder regression architecture

Numeric vectors do not use a token lookup table. A linear **input projection** maps
2 features to `d_model` learned features. Learned positional vectors add order.
Transformer blocks then mix information across all real timesteps.

To predict one value per sequence, **masked mean pooling** averages only real output
positions. A regression head maps the pooled vector to one scalar. Alternatives
include a special `[CLS]` token, max pooling, or attention pooling.

There is no causal attention mask. We pass only `src_key_padding_mask`.

Reference: [`TransformerEncoderLayer`](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html),
[`MSELoss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.MSELoss.html)"""),
]

REG_CELLS += exercise(
    "define bidirectional regression",
    "Complete input projection, positions, encoder, masked mean, and scalar head. Return shape `(B,)`.",
    """class SequenceRegressor(nn.Module):
    def __init__(self, n_features=2, d_model=64, n_head=4, n_layer=2, max_len=40):
        super().__init__()
        # TODO

    def forward(self, x, padding_mask):
        # TODO — importantly, no causal mask
        return None""",
    """try:
    reg = SequenceRegressor().to(device)
    pred = reg(px.to(device), pmask.to(device))
    assert pred.shape == (4,) and torch.isfinite(pred).all()
    print("🟢 Prediction shape:", pred.shape)
except Exception as exc:
    print("🟡 Finish the model, then rerun:", type(exc).__name__, exc)""",
)

REG_CELLS += [
    md("""## 5. Baselines and regression metrics

Always compare with a cheap baseline. “Predict the training target mean” has no
features and often exposes broken pipelines. Metrics in original target units:

- **MSE:** mean squared error; emphasizes large mistakes and is smooth to optimize.
- **RMSE:** square root of MSE; same units as target.
- **MAE:** mean absolute error; less dominated by outliers.
- **R²:** improvement over predicting the evaluation-set mean; 1 is perfect, 0 is
  no better, and negative is worse. It is not a percentage of predictions correct.

Optimize standardized-target MSE, but report all metrics after inverse scaling."""),
]

REG_CELLS += exercise(
    "implement metrics",
    "Given equal one-dimensional tensors `pred` and `actual`, return a dictionary containing mse, rmse, mae, and r2.",
    """def regression_metrics(pred, actual):
    return None""",
    """m = regression_metrics(torch.tensor([1.,2.,3.]), torch.tensor([1.,2.,4.]))
if m is None:
    print("🟡 Not attempted.")
else:
    assert abs(m["mse"] - 1/3) < 1e-6 and 0 <= m["r2"] <= 1
    print("🟢", m)""",
)

REG_CELLS += [
    md("""## 6. Training, early stopping, and diagnosis

An **epoch** is one pass through the training loader. Shuffle training examples each
epoch, but not validation/test. Early stopping saves the checkpoint with lowest
validation loss and stops after `patience` unimproved epochs. Test remains untouched.

TensorBoard records loss and learning rate. Residual plots later show whether errors
are centered randomly or reveal bias/nonlinearity/unequal variance."""),
    md("""# Answer key — complete runnable implementation

The independent `ans_` pipeline below includes splitting, scaling, collating,
training, checkpointing, test metrics, plots, and a direct masking experiment."""),
    code("""# Answers 1–3: split, train-only preprocessing, and collation
g = torch.Generator().manual_seed(SEED)
n_train = int(.70*len(examples)); n_val = int(.15*len(examples))
ans_train, ans_val, ans_test = random_split(examples, [n_train,n_val,len(examples)-n_train-n_val], generator=g)
train_x = torch.cat([x for x,y in ans_train], dim=0)
train_y = torch.stack([y for x,y in ans_train])
ans_xmean = train_x.mean(0); ans_xstd = train_x.std(0, correction=0).clamp_min(1e-6)
ans_ymean = train_y.mean(); ans_ystd = train_y.std(correction=0).clamp_min(1e-6)

def ans_collate(batch):
    xs, ys = zip(*batch)
    lengths = torch.tensor([len(x) for x in xs])
    padded = nn.utils.rnn.pad_sequence(xs, batch_first=True)
    mask = torch.arange(padded.size(1))[None,:] >= lengths[:,None]
    padded = (padded - ans_xmean) / ans_xstd
    targets = (torch.stack(ys) - ans_ymean) / ans_ystd
    return padded, targets, mask

train_loader = DataLoader(ans_train, batch_size=32, shuffle=True, collate_fn=ans_collate)
val_loader = DataLoader(ans_val, batch_size=64, collate_fn=ans_collate)
test_loader = DataLoader(ans_test, batch_size=64, collate_fn=ans_collate)
"""),
    code("""# Answer 4: bidirectional encoder and masked pooling
class AnswerSequenceRegressor(nn.Module):
    def __init__(self, n_features=2, d_model=64, n_head=4, n_layer=2, max_len=40):
        super().__init__()
        self.input_projection = nn.Linear(n_features, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_head, 4*d_model, .1,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layer, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, d_model//2), nn.GELU(), nn.Linear(d_model//2, 1))

    def encode(self, x, padding_mask):
        pos = torch.arange(x.size(1), device=x.device)
        h = self.input_projection(x) + self.position_embedding(pos)[None,:,:]
        return self.encoder(h, src_key_padding_mask=padding_mask)  # no causal mask

    def forward(self, x, padding_mask):
        h = self.norm(self.encode(x, padding_mask))
        valid = (~padding_mask).unsqueeze(-1).to(h.dtype)
        pooled = (h * valid).sum(1) / valid.sum(1).clamp_min(1)
        return self.head(pooled).squeeze(-1)

ans_model = AnswerSequenceRegressor().to(device)
print(f"Parameters: {sum(p.numel() for p in ans_model.parameters()):,}")
"""),
    code("""# Answer 5: metric implementation, collection, and mean baseline
def ans_metrics(pred, actual):
    error = pred - actual
    mse = (error**2).mean()
    return {"mse": mse.item(), "rmse": mse.sqrt().item(),
            "mae": error.abs().mean().item(),
            "r2": (1 - (error**2).sum()/((actual-actual.mean())**2).sum()).item()}

test_targets = torch.stack([y for x,y in ans_test])
baseline = torch.full_like(test_targets, ans_ymean)
print("Mean baseline:", ans_metrics(baseline, test_targets))
"""),
    code("""# Answer 6: explicit epoch loop with early stopping
def run_epoch(model, loader, optimizer=None):
    training = optimizer is not None
    model.train(training); total = count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for x, y, mask in loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            pred = model(x, mask); loss = F.mse_loss(pred, y)
            if training:
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            total += loss.item()*len(y); count += len(y)
    return total/count

epochs = 10 if FAST_MODE else 60; patience = 5
optimizer = torch.optim.AdamW(ans_model.parameters(), lr=1e-3, weight_decay=1e-3)
writer = SummaryWriter("runs/regression"); Path("checkpoints").mkdir(exist_ok=True)
history = {"train":[], "val":[]}; best = float("inf"); stale = 0
for epoch in range(epochs):
    tr = run_epoch(ans_model, train_loader, optimizer); va = run_epoch(ans_model, val_loader)
    history["train"].append(tr); history["val"].append(va)
    writer.add_scalars("standardized_mse", {"train":tr, "validation":va}, epoch)
    print(f"epoch {epoch+1:02d} | train {tr:.4f} | val {va:.4f}")
    if va < best:
        best=va; stale=0; torch.save(ans_model.state_dict(), "checkpoints/regression_best.pt")
    else:
        stale += 1
        if stale >= patience: print("early stopping"); break
writer.close()
"""),
    code("""# Final evaluation in original units and diagnostic plots
@torch.no_grad()
def collect_predictions(model, loader):
    model.eval(); ps=[]; ys=[]
    for x,y,mask in loader:
        p = model(x.to(device), mask.to(device)).cpu()
        ps.append(p*ans_ystd+ans_ymean); ys.append(y*ans_ystd+ans_ymean)
    return torch.cat(ps), torch.cat(ys)

ans_model.load_state_dict(torch.load("checkpoints/regression_best.pt", map_location=device, weights_only=True))
pred, actual = collect_predictions(ans_model, test_loader)
print("Transformer:", ans_metrics(pred, actual))
fig, ax = plt.subplots(1,3,figsize=(13,3.5))
ax[0].plot(history["train"],label="train"); ax[0].plot(history["val"],label="validation"); ax[0].set_title("Learning curves"); ax[0].legend()
ax[1].scatter(actual,pred,alpha=.7); lo=min(actual.min(),pred.min()); hi=max(actual.max(),pred.max()); ax[1].plot([lo,hi],[lo,hi],"k--"); ax[1].set(xlabel="actual",ylabel="predicted",title="Predictions")
ax[2].scatter(pred,pred-actual,alpha=.7); ax[2].axhline(0,color="k",ls="--"); ax[2].set(xlabel="predicted",ylabel="residual",title="Residuals")
plt.tight_layout(); plt.show()
"""),
    md("""## 7. Prove the mask distinction

This is the conceptual heart of the notebook. Change only the final real timestep
and compare the encoded representation at position 0. With bidirectional attention,
position 0 can change because it sees the future. With a correct causal mask it
would remain unchanged (in evaluation mode) because future positions are forbidden.

The padding mask still matters: changing a *padded* value should not affect pooled
predictions when masking is implemented correctly."""),
    code("""ans_model.eval()
x, y, mask = next(iter(test_loader)); x=x[:1].to(device); mask=mask[:1].to(device)
x_changed = x.clone(); last_real = int((~mask[0]).sum())-1; x_changed[0,last_real,0] += 5
with torch.no_grad():
    h1 = ans_model.encode(x,mask); h2 = ans_model.encode(x_changed,mask)
delta = (h1[0,0]-h2[0,0]).abs().mean().item()
print(f"Mean change at position 0 after changing a future token: {delta:.6f}")
assert delta > 1e-6
"""),
    code("""%load_ext tensorboard
%tensorboard --logdir runs/regression
""", ["skip-validation"]),
    md("""## Interpret and iterate

If the transformer does not beat the mean baseline, first verify inverse target
scaling, masks, and that optimizer updates occur. Then increase epochs. Training
loss far below validation suggests overfitting; add data, dropout, weight decay, or
reduce capacity. Structured residual curves suggest missing model flexibility or a
preprocessing issue.

Experiments:

1. Remove the trend term from data generation and compare difficulty.
2. Replace masked mean pooling with the final real timestep.
3. Add a causal mask and measure the result; is it intrinsically wrong, or merely
   an unnecessary restriction for this sequence-to-one task?
4. Compare a linear model or MLP on hand-engineered mean/trend features.
5. Increase noise to observe irreducible error.

Real regression tasks also require missing-data policies, distribution shift checks,
domain-informed splits, uncertainty, and fairness/safety evaluation."""),
    md("""## References

- PyTorch [`random_split`](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.random_split)
- PyTorch [`TransformerEncoder`](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html)
- PyTorch [`TransformerEncoderLayer.forward`](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html#torch.nn.TransformerEncoderLayer.forward)
- PyTorch [`MSELoss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.MSELoss.html)
- PyTorch [TensorBoard recipe](https://docs.pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html)

**Completion check:** explain why the scaler is train-only, how padding and causal
masks differ, why pooling is needed, what a negative R² means, and why the test set
is evaluated once."""),
]


def main() -> None:
    NOTEBOOKS.mkdir(exist_ok=True)
    outputs = {
        "01_shakespeare_causal_transformer.ipynb": notebook(TEXT_CELLS),
        "02_bidirectional_transformer_regression.ipynb": notebook(REG_CELLS),
    }
    for name, content in outputs.items():
        path = NOTEBOOKS / name
        path.write_text(json.dumps(content, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)} ({len(content['cells'])} cells)")


if __name__ == "__main__":
    main()
