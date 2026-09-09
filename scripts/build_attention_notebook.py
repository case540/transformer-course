"""Build notebook 03 without touching learner edits in notebooks 01 and 02."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "03_transformer_from_components.ipynb"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(source).strip() + "\n"}


def code(source: str, tags: list[str] | None = None) -> dict:
    return {
        "cell_type": "code", "execution_count": None,
        "metadata": {"tags": tags} if tags else {}, "outputs": [],
        "source": dedent(source).strip() + "\n",
    }


def exercise(title: str, prompt: str, starter: str, check: str) -> list[dict]:
    return [
        md(f"""### Exercise — {title}

{prompt}

Replace the `None` or `TODO`. The next cell checks the important behavior. If you
are stuck, inspect the shapes printed in the preceding example before viewing the
answer key at the bottom."""),
        code(starter, ["exercise"]), code(check, ["check"]),
    ]


cells = [
    md("""# 3 — A transformer, one tiny piece at a time

**Level:** advanced high school. You need ordinary algebra, averages, and basic
Python—not calculus.

**Goal:** understand and implement one small transformer block. We will not hide
attention inside `nn.Transformer`. Every important number will be visible.

By the end, you will be able to explain this pipeline:

`token IDs → embeddings + positions → Q, K, V → similarity scores → mask → softmax
→ weighted values → multiple heads → residual + LayerNorm → feed-forward network
→ residual + LayerNorm → predictions`

Notebook 1 uses these ideas to generate Shakespeare. Notebook 2 uses them for
regression. This notebook opens the box."""),
    md("""## How to work through this notebook

Run one cell at a time. Before running a numeric example, predict its **shape** and
whether each value should increase or decrease. Shapes catch more bugs than staring
at individual decimals.

Notation used throughout:

- `B`: batch size (sentences processed together)
- `T`: sequence length (tokens per sentence)
- `C` or `d_model`: vector width (features per token)
- `H`: number of attention heads
- `D`: width of one head, where `C = H × D`

We begin with `B=1`, `T=4`, and tiny vectors. Real models use the same operations
with much larger dimensions."""),
    code("""import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(12)
torch.set_printoptions(precision=3, sci_mode=False)
print("PyTorch", torch.__version__)
"""),
    md("""### Python/R bridge

- `x.shape` is like `dim(x)` in R. PyTorch dimensions start at `0`; R margins start
  at `1`.
- `@` is matrix multiplication, like `%*%` in R. `*` is element-by-element
  multiplication in both languages.
- `x.T` transposes a 2D matrix, like `t(x)` in R.
- `dim=-1` means “the final dimension.”
- `keepdim=True` keeps a reduced axis at size 1, similar to using `drop=FALSE` in R.
- A tensor is a multidimensional numeric array. A matrix is simply a 2D tensor."""),
    md("""## 1. Tokens and embeddings

Computers need numbers, so a tokenizer assigns each token an integer ID. The IDs
are labels, not quantities. ID 4 is not twice as meaningful as ID 2.

An **embedding table** stores one learned vector per token. Looking up a token ID
selects one row. During training, gradient descent adjusts those rows so useful
tokens develop useful features.

Our four-token sentence is `cats chase small mice`. We assign each token a made-up
4D embedding so the lookup is easy to inspect."""),
    code("""words = ["cats", "chase", "small", "mice"]
vocab = {word: i for i, word in enumerate(words)}
token_ids = torch.tensor([[vocab[word] for word in words]])  # shape (B=1, T=4)

embedding_table = nn.Embedding(num_embeddings=4, embedding_dim=4)
with torch.no_grad():
    embedding_table.weight.copy_(torch.tensor([
        [1.0, 0.2, 0.0, 0.1],   # cats
        [0.1, 1.0, 0.3, 0.0],   # chase
        [0.0, 0.2, 1.0, 0.2],   # small
        [0.9, 0.1, 0.1, 0.3],   # mice
    ]))

token_vectors = embedding_table(token_ids)
print("IDs:", token_ids)
print("Embedding-table shape:", tuple(embedding_table.weight.shape))
print("Looked-up vectors shape:", tuple(token_vectors.shape))
print(token_vectors)
"""),
]

cells += exercise(
    "look up token vectors",
    "Create IDs for `mice chase cats`, including a batch dimension, then use `embedding_table` to retrieve their vectors.",
    """practice_ids = None
practice_vectors = None""",
    """if practice_ids is None:
    print("🟡 Hint: [[vocab['mice'], vocab['chase'], vocab['cats']]]")
else:
    assert practice_ids.shape == (1, 3)
    assert practice_vectors.shape == (1, 3, 4)
    assert torch.equal(practice_vectors[0, 0], embedding_table.weight[vocab["mice"]])
    print("🟢 Correct shape and first lookup.")""",
)

cells += [
    md("""## 2. Position: the missing ingredient

Attention alone treats tokens like an unordered set. `cats chase mice` and `mice
chase cats` contain the same tokens but mean different things.

We therefore add a position vector to each token vector. Addition preserves shape:
`(B,T,C) + (1,T,C) → (B,T,C)`. Position vectors can be learned, as below, or
calculated using sine/cosine functions.

The resulting vector contains both **what** the token is and **where** it is."""),
    code("""position_table = nn.Embedding(num_embeddings=4, embedding_dim=4)
with torch.no_grad():
    position_table.weight.copy_(torch.tensor([
        [0.00, 0.00, 0.00, 0.00],
        [0.05, 0.00, 0.05, 0.00],
        [0.10, 0.00, 0.10, 0.00],
        [0.15, 0.00, 0.15, 0.00],
    ]))

positions = torch.arange(4)
x = token_vectors + position_table(positions)[None, :, :]
print("token_vectors:", token_vectors.shape)
print("position vectors after [None,:,:]:", position_table(positions)[None,:,:].shape)
print("combined x:", x.shape)
print(x)
"""),
    md("""### Broadcasting picture

The position tensor has one row of sequences and is reused for every batch item:

```text
tokens:     (B, T, C)
positions:  (1, T, C)
                 ↓ reused B times
result:     (B, T, C)
```

Inside an index, Python's `None` inserts a size-1 axis; it does **not** mean missing
data. `positions.unsqueeze(0)` does the same thing."""),
    md("""## 3. The attention idea before the equations

Suppose the token `chase` wants information about who is chasing and what is being
chased. It asks a question—a **query**. Every token advertises what it contains—a
**key**. A query-key match determines how much information to retrieve from each
token's **value**.

Library analogy:

- Query: the topic written on your request slip.
- Key: the label on each book.
- Value: the book's actual contents.

Queries and keys decide **where to look**. Values decide **what to bring back**.
All three are learned linear transformations of the current token vectors."""),
    md("""## 4. Linear transformations create Q, K, and V

A linear layer computes `output = input @ weight.T + bias`. Geometrically, it mixes,
rotates, stretches, and shifts features. Q, K, and V use separate weights because
“what I seek,” “how I should be found,” and “what I contribute” are different jobs.

For maximum visibility, this first example uses 2D inputs and hand-chosen matrices.
Bias is omitted."""),
    code("""tiny_x = torch.tensor([
    [1.0, 0.0],  # token A
    [0.0, 1.0],  # token B
    [1.0, 1.0],  # token C
])
W_Q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
W_K = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
W_V = torch.tensor([[1.0, 0.0], [1.0, 1.0]])

Q = tiny_x @ W_Q.T
K = tiny_x @ W_K.T
V = tiny_x @ W_V.T
print("x:\\n", tiny_x)
print("Q = x @ W_Q.T:\\n", Q)
print("K = x @ W_K.T:\\n", K)
print("V = x @ W_V.T:\\n", V)
"""),
]

cells += exercise(
    "compute one query",
    "Compute the query for token C (`tiny_x[2]`) using matrix multiplication and `W_Q.T`.",
    """query_c = None""",
    """if query_c is None:
    print("🟡 Use the same formula as Q, but select one input row.")
else:
    assert torch.allclose(query_c, Q[2])
    print("🟢 Query C:", query_c)""",
)

cells += [
    md("""## 5. Dot products measure query-key match

The dot product multiplies matching coordinates and adds them:

`[a,b] · [c,d] = a×c + b×d`

Large positive values mean alignment, values near zero mean little alignment, and
negative values mean opposing directions. Computing every query against every key
at once gives `scores = Q @ K.T`, a `(T,T)` matrix.

**Rows are queries; columns are keys.** Cell `[i,j]` asks: “How strongly does token
`i` match token `j`?”"""),
    code("""raw_scores = Q @ K.T
manual_c_to_b = Q[2,0]*K[1,0] + Q[2,1]*K[1,1]
print("Raw scores (rows=query, columns=key):\\n", raw_scores)
print("C→B by hand:", manual_c_to_b.item(), "| matrix entry:", raw_scores[2,1].item())
assert torch.isclose(manual_c_to_b, raw_scores[2,1])
"""),
    md("""## 6. Why divide by √D?

Longer random vectors tend to have larger dot products simply because more terms
are added. Very large scores make softmax nearly all-or-nothing, causing weak
learning signals. **Scaled dot-product attention** divides by `sqrt(D)`, where `D`
is the query/key width.

This does not change which score is largest; it makes the confidence less extreme."""),
    code("""D = Q.shape[-1]
scaled_scores = raw_scores / math.sqrt(D)
print("D:", D, "sqrt(D):", round(math.sqrt(D), 3))
print("Raw row 2:   ", raw_scores[2])
print("Scaled row 2:", scaled_scores[2])
"""),
    md("""## 7. Softmax turns scores into attention weights

Softmax converts each score row into positive weights that sum to 1:

`softmax(zᵢ) = exp(zᵢ) / Σ exp(zⱼ)`

You can read a row as percentages of attention. Softmax is applied across keys
(`dim=-1`, the columns), independently for each query."""),
    code("""weights = F.softmax(scaled_scores, dim=-1)
print("Attention weights:\\n", weights)
print("Each row sums to:", weights.sum(dim=-1))

fig, ax = plt.subplots(figsize=(4.5, 3.5))
image = ax.imshow(weights.detach(), cmap="Blues", vmin=0, vmax=1)
for i in range(3):
    for j in range(3): ax.text(j, i, f"{weights[i,j]:.2f}", ha="center", va="center")
ax.set(xlabel="key (look at)", ylabel="query (looking)", xticks=range(3), yticks=range(3),
       xticklabels=list("ABC"), yticklabels=list("ABC"), title="One attention head")
plt.colorbar(image, ax=ax); plt.show()
"""),
]

cells += exercise(
    "implement scaled score weights",
    "Given Q and K, compute scaled dot-product scores and softmax across keys. Do not use PyTorch's built-in attention function.",
    """def attention_weights(q, k):
    # TODO
    return None""",
    """practice_weights = attention_weights(Q, K)
if practice_weights is None:
    print("🟡 Formula: softmax((q @ k.T) / sqrt(last_dimension)).")
else:
    assert practice_weights.shape == (3,3)
    assert torch.allclose(practice_weights.sum(-1), torch.ones(3))
    assert torch.allclose(practice_weights, weights)
    print("🟢 Scaled attention weights are correct.")""",
)

cells += [
    md("""## 8. Weighted values produce the output

Weights alone are not the output. Each query takes a weighted average of **value
vectors**: `output = weights @ V`.

For query A, if weights are `[0.40, 0.20, 0.40]`, its output is
`0.40×V_A + 0.20×V_B + 0.40×V_C`. Every output can now contain information from
every input token."""),
    code("""attention_output = weights @ V
manual_a = sum(weights[0,j] * V[j] for j in range(3))
print("Values:\\n", V)
print("Attention output:\\n", attention_output)
print("A output by weighted sum:", manual_a)
assert torch.allclose(manual_a, attention_output[0])
"""),
]

cells += exercise(
    "write single-head attention",
    "Write a function that returns both output and weights for already-projected Q, K, and V.",
    """def simple_attention(q, k, v):
    # TODO
    return None, None""",
    """practice_out, practice_w = simple_attention(Q, K, V)
if practice_out is None:
    print("🟡 Reuse scaled scores → softmax → weights @ v.")
else:
    assert torch.allclose(practice_out, attention_output)
    assert torch.allclose(practice_w, weights)
    print("🟢 Single-head attention matches the hand-built example.")""",
)

cells += [
    md("""## 9. Masks control where attention may look

A **causal mask** prevents a token from seeing future tokens during next-token
prediction. We replace forbidden scores with negative infinity. Softmax turns
`exp(-∞)` into zero weight.

```text
query 0 may see key:  0
query 1 may see keys: 0 1
query 2 may see keys: 0 1 2
```

A padding mask is different: it hides empty filler positions. Notebook 2 uses a
padding mask but no causal mask because regression may use the whole sequence."""),
    code("""causal_mask = torch.triu(torch.ones(3,3, dtype=torch.bool), diagonal=1)
masked_scores = scaled_scores.masked_fill(causal_mask, float("-inf"))
causal_weights = F.softmax(masked_scores, dim=-1)
print("Mask (True = forbidden):\\n", causal_mask)
print("Scores after masking:\\n", masked_scores)
print("Causal weights:\\n", causal_weights)
assert torch.all(causal_weights[causal_mask] == 0)
"""),
]

cells += exercise(
    "add optional causal masking",
    "Extend attention so `causal=True` prevents looking above the score matrix diagonal. Return output and weights.",
    """def masked_attention(q, k, v, causal=False):
    # TODO
    return None, None""",
    """masked_out, masked_w = masked_attention(Q, K, V, causal=True)
if masked_out is None:
    print("🟡 Create a bool upper triangle and use masked_fill before softmax.")
else:
    assert torch.all(masked_w[causal_mask] == 0)
    assert torch.allclose(masked_w.sum(-1), torch.ones(3))
    assert torch.allclose(masked_w[0], torch.tensor([1.,0.,0.]))
    print("🟢 Future attention is exactly zero.")""",
)

cells += [
    md("""## 10. Batch dimensions and the transpose trap

Real input has shape `(B,T,C)`, not just `(T,C)`. To compare every query with every
key, transpose only the final two axes:

```text
Q:                  (B,T,D)
K.transpose(-2,-1): (B,D,T)
Q @ Kᵀ:             (B,T,T)
```

Do not use `.T` on tensors with more than two dimensions. `transpose(-2,-1)` states
exactly which axes to swap."""),
    code("""batch_q = Q.unsqueeze(0).repeat(2,1,1)  # two examples
batch_k = K.unsqueeze(0).repeat(2,1,1)
batch_scores = batch_q @ batch_k.transpose(-2,-1)
print("Q batch:", batch_q.shape)
print("transposed K:", batch_k.transpose(-2,-1).shape)
print("scores:", batch_scores.shape)
"""),
    md("""## 11. Why multiple heads?

One attention head creates one kind of query-key matching system. Multiple heads
can learn different relationships—perhaps one tracks nearby adjectives while
another tracks subject/verb relationships.

For `C=4` and `H=2`, each head has `D=2` features. We reshape `(B,T,C)` into
`(B,T,H,D)`, move heads before time to get `(B,H,T,D)`, run attention independently,
then reverse the reshape and mix heads with an output projection.

Splitting is not enough by itself: learned Q/K/V projections make each head view the
input differently."""),
    code("""B, T, C = 1, 4, 4
H = 2
D = C // H
demo = torch.arange(B*T*C).reshape(B,T,C)
split = demo.reshape(B,T,H,D)
heads_first = split.transpose(1,2)
joined = heads_first.transpose(1,2).contiguous().reshape(B,T,C)
print("original", demo.shape, "→ split", split.shape, "→ heads first", heads_first.shape)
print("joined", joined.shape, "| exactly recovered:", torch.equal(joined, demo))
"""),
]

cells += exercise(
    "split into heads",
    "Implement the reshape and transpose from `(B,T,C)` to `(B,H,T,D)`. Check that `C` divides evenly by `n_heads`.",
    """def split_heads(tensor, n_heads):
    # TODO
    return None""",
    """practice_heads = split_heads(demo, 2)
if practice_heads is None:
    print("🟡 First reshape to (B,T,H,D), then transpose axes 1 and 2.")
else:
    assert practice_heads.shape == (1,2,4,2)
    assert torch.equal(practice_heads, heads_first)
    print("🟢 Head shape:", practice_heads.shape)""",
)

cells += [
    md("""## 12. Build multi-head causal self-attention

**Self-attention** means Q, K, and V all come from the same sequence. “Self” does
not mean the token only attends to itself.

This module combines four learned linear layers:

1. one projection each for Q, K, V;
2. reshape into heads;
3. scaled dot products, mask, softmax, weighted values;
4. join heads and apply an output projection.

Dropout is omitted to keep this teaching implementation deterministic."""),
    code("""class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, causal=True):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.causal = causal
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def _split(self, tensor):
        B, T, C = tensor.shape
        return tensor.reshape(B,T,self.n_heads,self.head_dim).transpose(1,2)

    def forward(self, x, return_weights=False):
        B, T, C = x.shape
        q, k, v = self._split(self.q_proj(x)), self._split(self.k_proj(x)), self._split(self.v_proj(x))
        scores = q @ k.transpose(-2,-1) / math.sqrt(self.head_dim)
        if self.causal:
            mask = torch.triu(torch.ones(T,T,dtype=torch.bool,device=x.device), diagonal=1)
            scores = scores.masked_fill(mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        heads = weights @ v
        joined = heads.transpose(1,2).contiguous().reshape(B,T,C)
        output = self.out_proj(joined)
        return (output, weights) if return_weights else output

attention = MultiHeadSelfAttention(d_model=4, n_heads=2, causal=True)
attended, learned_weights = attention(x, return_weights=True)
print("input:", x.shape, "output:", attended.shape, "weights:", learned_weights.shape)
print("Weights for head 0:\\n", learned_weights[0,0].detach())
"""),
    md("""## 13. Residual connections preserve a direct path

Attention returns an update, but replacing the input completely would make deep
networks hard to optimize. A **residual connection** adds the old representation:

`x_after_attention = x + attention(x)`

If the learned update is temporarily poor, the original information still has a
direct route. Shapes must match exactly."""),
    code("""after_residual = x + attended
print("x:", x.shape, "+ attention update:", attended.shape, "=", after_residual.shape)
print("First token before:", x[0,0])
print("Update:", attended[0,0])
print("After addition:", after_residual[0,0])
"""),
    md("""## 14. Layer normalization keeps values manageable

`LayerNorm(C)` normalizes the `C` features of each token separately, then applies
learned scale and shift parameters. It helps stabilize deep networks.

For each token vector it approximately performs:

`normalized = (x - mean) / sqrt(variance + ε)`

Unlike a simple standardization, LayerNorm's scale and shift are trainable."""),
    code("""norm = nn.LayerNorm(4)
normalized = norm(after_residual)
print("Before, first token:", after_residual[0,0])
print("After, first token: ", normalized[0,0])
print("Mean:", normalized[0,0].mean().item(), "variance:", normalized[0,0].var(correction=0).item())
"""),
    md("""## 15. The feed-forward network thinks at each position

Attention communicates **between tokens**. The feed-forward network (FFN) transforms
features **within each token** using the same small neural network at every position:

`C → 4C → GELU → C`

The expansion gives the model room to build richer combinations. GELU is a smooth
nonlinearity; without nonlinearities, stacking linear layers would collapse into
one linear transformation."""),
    code("""ffn = nn.Sequential(
    nn.Linear(4, 16),
    nn.GELU(),
    nn.Linear(16, 4),
)
ffn_update = ffn(normalized)
print("Input and output shapes match:", normalized.shape, ffn_update.shape)
print("The same FFN processes all", normalized.shape[1], "token positions.")
"""),
    md("""## 16. Assemble one complete transformer block

We use **pre-normalization**, common in modern models:

```text
x ──┬──────────────────────────────► + ──► x
    └─► LayerNorm ─► Attention ─────┘
x ──┬──────────────────────────────► + ──► output
    └─► LayerNorm ─► FFN ───────────┘
```

Both sublayers get residual paths. A transformer model stacks several blocks."""),
]

cells += exercise(
    "assemble a transformer block",
    "Define two LayerNorms, our attention module, and an FFN. In `forward`, implement the two pre-normalized residual updates.",
    """class TransformerBlock(nn.Module):
    def __init__(self, d_model=4, n_heads=2):
        super().__init__()
        # TODO

    def forward(self, x):
        # TODO
        return None""",
    """try:
    practice_block = TransformerBlock()
    practice_result = practice_block(x)
    assert practice_result.shape == x.shape
    assert torch.isfinite(practice_result).all()
    print("🟢 Block preserves (B,T,C):", practice_result.shape)
except Exception as exc:
    print("🟡 Complete both residual sublayers. Current issue:", type(exc).__name__, exc)""",
)

cells += [
    md("""## 17. From a block to a tiny language model

A language model adds four outer pieces:

1. token embeddings;
2. position embeddings;
3. one or more transformer blocks;
4. a final linear layer producing one **logit** per vocabulary token.

Logits are unrestricted scores. Softmax converts them to probabilities. During
training, cross-entropy compares logits at every position with the true next token."""),
    md("""# Answer key and complete implementation

These answers use separate `answer_` names, so this section runs even if exercise
cells are unfinished."""),
    code("""# Answers: lookup, scaled attention, masking, and head splitting
answer_ids = torch.tensor([[vocab["mice"], vocab["chase"], vocab["cats"]]])
answer_vectors = embedding_table(answer_ids)

def answer_attention(q, k, v, causal=False):
    scores = q @ k.transpose(-2,-1) / math.sqrt(q.shape[-1])
    if causal:
        T = q.shape[-2]
        mask = torch.triu(torch.ones(T,T,dtype=torch.bool,device=q.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    return weights @ v, weights

def answer_split_heads(tensor, n_heads):
    B,T,C = tensor.shape
    if C % n_heads: raise ValueError("C must divide evenly by n_heads")
    return tensor.reshape(B,T,n_heads,C//n_heads).transpose(1,2)

assert answer_vectors.shape == (1,3,4)
assert torch.allclose(answer_attention(Q,K,V)[0], attention_output)
assert answer_split_heads(demo,2).shape == (1,2,4,2)
print("🟢 Foundational answers pass.")
"""),
    code("""# Answer: complete pre-normalized block
class AnswerTransformerBlock(nn.Module):
    def __init__(self, d_model=4, n_heads=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadSelfAttention(d_model, n_heads, causal=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model,4*d_model), nn.GELU(), nn.Linear(4*d_model,d_model))

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

answer_block = AnswerTransformerBlock()
block_output = answer_block(x)
print("Block:", x.shape, "→", block_output.shape)
"""),
    code("""# Complete tiny causal language model
class TinyLanguageModel(nn.Module):
    def __init__(self, vocab_size, max_len=8, d_model=16, n_heads=4, n_layers=2):
        super().__init__()
        self.max_len = max_len
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.blocks = nn.Sequential(*[AnswerTransformerBlock(d_model,n_heads) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids, targets=None):
        B,T = token_ids.shape
        positions = torch.arange(T, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)[None,:,:]
        x = self.blocks(x)
        logits = self.output(self.final_norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1,logits.shape[-1]), targets.reshape(-1))
        return logits, loss

model = TinyLanguageModel(vocab_size=8)
sample_ids = torch.tensor([[0,1,2,3]])
sample_targets = torch.tensor([[1,2,3,4]])
logits, loss = model(sample_ids, sample_targets)
print("Input IDs:", sample_ids.shape)
print("Logits:", logits.shape, "= (B,T,vocab_size)")
print("Initial loss:", round(loss.item(),3))
assert logits.shape == (1,4,8) and loss.ndim == 0
"""),
    md("""## 18. One training update, slowed down

The model starts random. One update does this:

1. **Forward:** calculate logits and loss.
2. **Clear gradients:** remove gradients from the previous update.
3. **Backward:** calculate how each parameter influenced loss.
4. **Step:** slightly change parameters to reduce loss.

A single step need not lower the current batch's loss every time in real training,
but repeatedly learning one tiny batch should clearly lower it. This is an
**overfit-one-batch test**, a valuable pipeline sanity check—not a useful final
model."""),
    code("""optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)
losses = []
for step in range(60):
    _, loss = model(sample_ids, sample_targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

plt.plot(losses); plt.xlabel("update"); plt.ylabel("cross-entropy loss")
plt.title("Overfitting one tiny example (a sanity check)"); plt.show()
print(f"Loss: {losses[0]:.3f} → {losses[-1]:.3f}")
assert losses[-1] < losses[0] * 0.2
"""),
    md("""## 19. Inspect learned attention carefully

Attention weights are useful for seeing where information flows, but they are not a
perfect explanation of a model's decision. Value vectors, residual paths, later
layers, and the output head also matter.

Each head gets its own `(T,T)` heatmap. Because this model is causal, every cell
above the diagonal must remain zero."""),
    code("""model.eval()
with torch.no_grad():
    positions = torch.arange(sample_ids.shape[1])
    embedded = model.token_embedding(sample_ids) + model.position_embedding(positions)[None,:,:]
    _, final_weights = model.blocks[0].attention(model.blocks[0].norm1(embedded), return_weights=True)

fig, axes = plt.subplots(1, final_weights.shape[1], figsize=(4*final_weights.shape[1],3))
for head, ax in enumerate(np.atleast_1d(axes)):
    matrix = final_weights[0,head]
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4): ax.text(j,i,f"{matrix[i,j]:.2f}",ha="center",va="center")
    ax.set(title=f"Head {head}", xlabel="key", ylabel="query", xticks=range(4), yticks=range(4))
plt.tight_layout(); plt.show()
"""),
    md("""## Common mistakes and how to diagnose them

- **Wrong softmax axis:** every query row must sum to 1.
- **Forgot scaling:** training may become overly confident or unstable as `D` grows.
- **Mask after softmax:** masking must happen before softmax, or remaining weights no
  longer form the intended distribution.
- **Wrong transpose:** attention scores should end in `(T,T)`.
- **Lost batch/head axis:** print shapes after every reshape.
- **`d_model` not divisible by heads:** each head needs an equal width.
- **No positions:** the model cannot naturally distinguish token order.
- **No nonlinearity:** stacked linear layers are still only a linear transformation.
- **No residual path:** deep optimization becomes harder.
- **Interpreting token IDs numerically:** IDs only select embedding rows.

When debugging, test invariants: finite values, expected shapes, attention rows sum
to 1, masked weights equal 0, and loss decreases on one tiny batch."""),
    md("""## Final mental model

A transformer block alternates two jobs:

1. **Attention:** each token gathers relevant information from allowed tokens.
2. **Feed-forward network:** each token processes its newly gathered information.

Residual connections preserve old information, LayerNorm stabilizes values, and
stacking blocks repeats communication and processing. The model learns the Q/K/V,
embedding, FFN, and output weights from examples; programmers specify the wiring.

### Explain these aloud

1. Why are token IDs not meaningful numbers?
2. What different jobs do Q, K, and V perform?
3. Why is the score matrix `(T,T)`?
4. Why divide scores by `sqrt(D)`?
5. Why do softmax rows sum to 1?
6. Why does a causal mask use negative infinity?
7. What can two heads do that one head may not?
8. What do residual connections preserve?
9. Why does the FFN need a nonlinearity?
10. Which pieces contain learned parameters?"""),
    md("""## References

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- PyTorch [`nn.Linear`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html)
- PyTorch [`softmax`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html)
- PyTorch [`LayerNorm`](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)
- PyTorch [`Embedding`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html)
- PyTorch [`scaled_dot_product_attention`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)

PyTorch's optimized attention API performs the same central score → mask → softmax
→ weighted-value calculation. We wrote it explicitly here so no step was hidden.

This notebook does not use TensorBoard because it focuses on a single block and a
60-step sanity check. Notebooks 1 and 2 demonstrate full training dashboards."""),
]


for index, cell in enumerate(cells):
    cell["id"] = f"cell-{index:03d}"

notebook = {
    "cells": cells,
    "metadata": {
        "colab": {"name": "Transformer from components", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}

OUTPUT.parent.mkdir(exist_ok=True)
OUTPUT.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
print(f"wrote {OUTPUT.relative_to(ROOT)} ({len(cells)} cells)")
