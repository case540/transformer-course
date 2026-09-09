"""Build the intentionally simple transformer lesson (Notebook 03 only)."""
from __future__ import annotations
import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "03_transformer_from_components.ipynb"

def md(s): return {"cell_type":"markdown","metadata":{},"source":dedent(s).strip()+"\n"}
def code(s, tags=None): return {"cell_type":"code","execution_count":None,"metadata":{"tags":tags} if tags else {},"outputs":[],"source":dedent(s).strip()+"\n"}
def exercise(title, prompt, starter, check):
    return [md(f"""### Your turn — {title}

{prompt}

Change only the `None` line. The next cell checks your answer. Answers are at the bottom."""), code(starter,["exercise"]), code(check,["check"])]

cells = [
md("""# A tiny transformer you can see

No giant matrices. No multi-head attention. No mysterious Q/K/V projections.

We will use one three-word sentence and **2D vectors that we can draw**. For our first transformer:

## `Q = K = V = the embedding vectors`

That is valid self-attention. Real transformers later learn separate Q, K, and V projections for more flexibility, but we do not need that complication to understand attention.

By the end, you will implement:

```text
word → embedding → attention → add old information → tiny neural network
```

The whole lesson runs in seconds."""),
md("""## The entire lesson in one picture

```text
       ┌────────────┐
words ─► embeddings ├───────────────┐
       └─────┬──────┘               │ keep the original
             ▼                      │
       ┌────────────┐               │
       │ attention  │               │
       │ Q = K = V  │               │
       └─────┬──────┘               │
             ▼                      ▼
       new information ───────────► +
                                    │
                                    ▼
                         tiny feed-forward network
                                    │
                                    ▼
                           transformed vectors
```

Attention lets words exchange information. The feed-forward network lets each word process what it received."""),
code("""import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(7)
torch.set_printoptions(precision=3, sci_mode=False)
print("PyTorch", torch.__version__)"""),
md("""## 1. What is an embedding?

An embedding is just a short list of numbers used to represent an item.

We have **six words**, but each word gets a **2-number vector**. The embedding table is therefore `6 × 2`, not square:

```text
                 feature 1   feature 2
king                2.0         1.8
queen               1.9         2.0
apple              -1.8         0.2
orange             -1.7         0.4
dog                 0.1        -1.8
cat                 0.3        -1.7
```

- Rows = vocabulary size = 6 words.
- Columns = embedding size = 2 features.

These numbers are hand-picked so we can draw them. A real model learns them from data."""),
code("""words = ["king", "queen", "apple", "orange", "dog", "cat"]
word_to_id = {word: i for i, word in enumerate(words)}
embedding_table = torch.tensor([
    [ 2.0,  1.8], [ 1.9,  2.0],
    [-1.8,  0.2], [-1.7,  0.4],
    [ 0.1, -1.8], [ 0.3, -1.7],
])
print("Words:", len(words))
print("Numbers per word:", embedding_table.shape[1])
print("Table shape:", tuple(embedding_table.shape), "← 6 words × 2 features")"""),
code("""plt.figure(figsize=(6,5))
plt.axhline(0,color="lightgray"); plt.axvline(0,color="lightgray")
for word, point in zip(words, embedding_table):
    plt.scatter(point[0],point[1],s=90)
    plt.annotate(word,point,xytext=(6,6),textcoords="offset points",fontsize=12)
plt.xlim(-2.4,2.5); plt.ylim(-2.3,2.5); plt.grid(alpha=.2)
plt.xlabel("embedding feature 1"); plt.ylabel("embedding feature 2")
plt.title("Similar words are near one another"); plt.show()"""),
md("""The axes do not have simple names like “royalty” or “animal.” Usually, individual coordinates are not interpretable. What matters is geometry: `king` is near `queen`, `apple` is near `orange`, and `dog` is near `cat`.

### Python/R bridge

`embedding_table[5]` selects the **sixth** row because Python starts at 0. The closest R selection is `embedding_table[6, ]`."""),
]
cells += exercise("look up `cat`", "Use `word_to_id` to find its row, then select that row from `embedding_table`.", "cat_vector = None", """if cat_vector is None: print("🟡 Try: embedding_table[word_to_id[...]]")
else:
    assert torch.equal(cat_vector,torch.tensor([0.3,-1.7])); print("🟢 cat →",cat_vector.tolist())""")
cells += [
md("""## 2. See embeddings from a real pretrained model

Our six vectors were invented. This optional cell loads **GloVe**, trained on Wikipedia text, and displays some real 50-dimensional word vectors.

Humans cannot draw 50 dimensions, so PCA makes a 2D “shadow.” It loses information: closeness in the picture is suggestive, not proof of identical meaning.

The first run downloads about 70 MB and caches it. Set the switch to `False` when offline; nothing else depends on GloVe."""),
code("""RUN_PRETRAINED_DEMO = True
if RUN_PRETRAINED_DEMO:
    try:
        import gensim.downloader as api
        glove = api.load("glove-wiki-gigaword-50")
        real_words = ["king","queen","prince","princess","man","woman",
                      "dog","cat","puppy","kitten","apple","orange","fruit"]
        vectors = np.stack([glove[word] for word in real_words])
        centered = vectors - vectors.mean(axis=0,keepdims=True)
        _,_,directions = np.linalg.svd(centered,full_matrices=False)
        points = centered @ directions[:2].T
        plt.figure(figsize=(8,6))
        for indexes,color in zip([range(0,6),range(6,10),range(10,13)],["purple","tab:blue","tab:orange"]):
            for i in indexes:
                plt.scatter(*points[i],color=color,s=70)
                plt.annotate(real_words[i],points[i],xytext=(5,5),textcoords="offset points")
        plt.axhline(0,color="lightgray"); plt.axvline(0,color="lightgray"); plt.grid(alpha=.2)
        plt.xlabel("PCA direction 1"); plt.ylabel("PCA direction 2")
        plt.title("2D view of real 50D GloVe embeddings"); plt.show()
    except Exception as error:
        print("Pretrained demo skipped:",error)
        print("All hand-made examples below still work.")
else: print("Skipped. Change RUN_PRETRAINED_DEMO to True when online.")""",["optional-download"]),
md("""## 3. Our tiny sentence

We now use three 2D embeddings:

```text
red    → [1.0, 0.0]
apple  → [0.8, 0.2]
tasty  → [0.0, 1.0]
```

`red` and `apple` point similarly, so their dot product will be large. `red` and `tasty` point in different directions, so their dot product will be zero.

The matrix shape is `3 words × 2 features`, or `(T=3, C=2)`."""),
code("""sentence = ["red","apple","tasty"]
X = torch.tensor([[1.0,0.0],[0.8,0.2],[0.0,1.0]])
plt.figure(figsize=(5,5)); plt.axhline(0,color="lightgray"); plt.axvline(0,color="lightgray")
for word,vector in zip(sentence,X):
    plt.arrow(0,0,vector[0],vector[1],head_width=.04,length_includes_head=True)
    plt.text(vector[0]+.04,vector[1]+.04,word,fontsize=12)
plt.xlim(-.1,1.2); plt.ylim(-.1,1.2); plt.grid(alpha=.2)
plt.xlabel("feature 1"); plt.ylabel("feature 2"); plt.title("Our three word vectors"); plt.show()"""),
md("""## 4. Q = K = V = X

- **Query (Q):** the vector doing the looking.
- **Key (K):** the vectors it compares itself with.
- **Value (V):** the vectors whose information it collects.

For now:

```python
Q = X
K = X
V = X
```

Each word asks: **“Which embedding vectors point in a direction similar to mine?”**"""),
code("""Q = X
K = X
V = X
print("Q = K = V =\\n",X)"""),
md("""## 5. Step one: compare every word with every word

The **dot product** compares two vectors:

```text
[a,b] dot [c,d] = a×c + b×d

red dot apple = [1,0] dot [0.8,0.2]
              = 1×0.8 + 0×0.2 = 0.8
```

`Q @ K.T` calculates all nine comparisons. In R, `@` corresponds to `%*%` and `.T` to `t()`."""),
code("""scores = Q @ K.T
print("Similarity scores:\\n",scores)
fig,ax=plt.subplots(figsize=(5,4)); image=ax.imshow(scores,cmap="Purples")
for r in range(3):
    for c in range(3): ax.text(c,r,f"{scores[r,c]:.2f}",ha="center",va="center",fontsize=12)
ax.set(xticks=range(3),yticks=range(3),xticklabels=sentence,yticklabels=sentence,
       xlabel="word being looked at",ylabel="word doing the looking",title="Dot-product similarity")
plt.colorbar(image,ax=ax); plt.show()"""),
]
cells += exercise("calculate one dot product", "Calculate `apple · tasty` by multiplying and adding their two coordinates.", "apple_dot_tasty = None", """if apple_dot_tasty is None: print("🟡 apple is X[1]; tasty is X[2].")
else:
    assert abs(float(apple_dot_tasty)-0.2)<1e-6; print("🟢 apple · tasty =",float(apple_dot_tasty))""")
cells += [
md("""## 6. Step two: turn scores into percentages

**Softmax** turns each row into positive numbers that add to 1—attention percentages.

First divide by `sqrt(embedding size)`. This keeps scores from becoming too extreme for large vectors:

```text
scaled scores = scores / √2
weights       = softmax(scaled scores)
```

Each word gets its own row of percentages."""),
code("""scaled_scores = scores / math.sqrt(2)
weights = F.softmax(scaled_scores,dim=1)
print("Attention weights:\\n",weights)
print("Row sums:",weights.sum(dim=1))
fig,ax=plt.subplots(figsize=(5,4)); ax.imshow(weights,cmap="Blues",vmin=0,vmax=1)
for r in range(3):
    for c in range(3): ax.text(c,r,f"{100*weights[r,c]:.0f}%",ha="center",va="center",fontsize=12)
ax.set(xticks=range(3),yticks=range(3),xticklabels=sentence,yticklabels=sentence,
       xlabel="gets information from",ylabel="word being updated",title="Attention percentages"); plt.show()"""),
md("""Read across a row. The `red` row shows how `red` divides 100% of its attention among `red`, `apple`, and `tasty`.

**Debugging rule:** every row must add to 1. If not, softmax probably used the wrong dimension."""),
]
cells += exercise("make attention percentages", "Use `F.softmax` on `scaled_scores`. Choose the dimension that makes each row sum to 1.", "my_weights = None", """if my_weights is None: print("🟡 Rows are dimension 1 in this matrix.")
else:
    assert torch.allclose(my_weights,weights); print("🟢 Row sums:",my_weights.sum(1).tolist())""")
cells += [
md("""## 7. Step three: mix the value vectors

Each word takes a weighted average of the value vectors:

```text
output = attention percentages @ V
```

Remember, `V = X`. If `red` pays attention to `apple`, its new vector receives some of `apple`'s information."""),
code("""attention_output = weights @ V
print("Old embeddings:\\n",X)
print("New context-aware vectors:\\n",attention_output)
red_by_hand = weights[0,0]*V[0] + weights[0,1]*V[1] + weights[0,2]*V[2]
print("red by hand:",red_by_hand)
assert torch.allclose(red_by_hand,attention_output[0])"""),
md("""## Attention is only three lines

```python
scores = X @ X.T / sqrt(embedding_size)
weights = softmax(scores)
output = weights @ X
```

The first `@` asks **where should I look?**

Softmax answers **what percentage goes to each place?**

The second `@` answers **what information do I receive?**"""),
]
cells += exercise("implement simple self-attention", "Complete the three-line function. Use the embeddings as Q, K, and V.", """def simple_self_attention(embeddings):
    # TODO
    return None, None""", """my_output,my_attention=simple_self_attention(X)
if my_output is None: print("🟡 Copy the three-line recipe above.")
else:
    assert torch.allclose(my_attention,weights) and torch.allclose(my_output,attention_output)
    print("🟢 Your function matches every value.")""")
cells += [
md("""## 8. Optional rule: do not look into the future

When generating text, a word must not peek at later words. A **causal mask** says:

```text
red    may see: red
apple  may see: red, apple
tasty  may see: red, apple, tasty
```

Replace forbidden scores with negative infinity **before softmax**. They become exactly 0%."""),
code("""future_is_forbidden = torch.tensor([[False,True,True],[False,False,True],[False,False,False]])
masked_scores = scaled_scores.masked_fill(future_is_forbidden,float("-inf"))
causal_weights = F.softmax(masked_scores,dim=1)
print("True means hidden:\\n",future_is_forbidden)
fig,ax=plt.subplots(figsize=(5,4)); ax.imshow(causal_weights,cmap="Blues",vmin=0,vmax=1)
for r in range(3):
    for c in range(3): ax.text(c,r,f"{100*causal_weights[r,c]:.0f}%",ha="center",va="center")
ax.set(xticks=range(3),yticks=range(3),xticklabels=sentence,yticklabels=sentence,
       xlabel="gets information from",ylabel="word being updated",title="Causal attention"); plt.show()"""),
md("""The first word gives 100% attention to itself because it cannot see anything else. The last word sees all three.

Classification or regression over a complete sentence often does **not** use this mask. Notebook 2 is an example."""),
md("""## 9. Make a tiny transformer block

A transformer also:

1. adds the original embeddings back (**residual connection**);
2. normalizes the numbers (**LayerNorm**);
3. runs a tiny neural network on each word (**feed-forward network**).

```text
X ───────────────┐
│                ▼
└─ attention ───► + ─► LayerNorm ─► tiny neural network ─► output
```

The residual keeps old information. The neural network processes the new context."""),
code("""class SimplestTransformerBlock(nn.Module):
    def __init__(self,embedding_size=2):
        super().__init__()
        self.norm=nn.LayerNorm(embedding_size)
        self.feed_forward=nn.Sequential(nn.Linear(embedding_size,4),nn.ReLU(),nn.Linear(4,embedding_size))
    def forward(self,embeddings):
        # Q = K = V = embeddings
        scores=embeddings @ embeddings.T / math.sqrt(embeddings.shape[-1])
        weights=F.softmax(scores,dim=1)
        attended=weights @ embeddings
        mixed=self.norm(embeddings+attended)  # residual + normalization
        output=mixed+self.feed_forward(mixed)
        return output,weights

block=SimplestTransformerBlock()
transformed,block_weights=block(X)
print("Input shape:",tuple(X.shape),"Output shape:",tuple(transformed.shape))
print(transformed)"""),
md("""### What is learned here?

- LayerNorm learns a scale and shift.
- The feed-forward network learns weights and biases.
- If `X` came from `nn.Embedding`, embeddings would be learned too.
- Attention has no Q/K/V weight matrices because we deliberately chose `Q=K=V=X`.

This simpler-than-standard block genuinely computes self-attention."""),
md("""## 10. Optional: one step toward a standard transformer

A standard transformer learns different versions of Q, K, and V:

```python
Q = query_layer(X)
K = key_layer(X)
V = value_layer(X)
```

Each remains a 2D vector. **Nothing else changes.** Skip this on your first pass."""),
code("""query_layer=nn.Linear(2,2,bias=False)
key_layer=nn.Linear(2,2,bias=False)
value_layer=nn.Linear(2,2,bias=False)
learned_Q,learned_K,learned_V=query_layer(X),key_layer(X),value_layer(X)
learned_weights=F.softmax(learned_Q @ learned_K.T / math.sqrt(2),dim=1)
learned_output=learned_weights @ learned_V
print("Q/K/V shapes:",learned_Q.shape,learned_K.shape,learned_V.shape)
print("Output shape:",learned_output.shape)"""),
md("""## Answer key

Try the four exercises first."""),
code("""answer_cat_vector=embedding_table[word_to_id["cat"]]
answer_apple_dot_tasty=X[1,0]*X[2,0]+X[1,1]*X[2,1]
answer_weights=F.softmax(scaled_scores,dim=1)
def answer_self_attention(embeddings):
    scores=embeddings @ embeddings.T / math.sqrt(embeddings.shape[-1])
    weights=F.softmax(scores,dim=1)
    return weights @ embeddings,weights
answer_output,answer_attention=answer_self_attention(X)
assert torch.equal(answer_cat_vector,torch.tensor([0.3,-1.7]))
assert abs(answer_apple_dot_tasty.item()-0.2)<1e-6
assert torch.allclose(answer_weights.sum(1),torch.ones(3))
assert torch.allclose(answer_output,attention_output)
print("🟢 All answers pass.")"""),
md("""## Things to remember

1. An embedding is a vector. Vocabulary size and embedding size are unrelated.
2. Here, `Q = K = V = embeddings`.
3. Dot products measure similarity.
4. Softmax turns scores into percentages summing to 1.
5. Multiplying percentages by V mixes information.
6. A causal mask makes future attention 0%.
7. A residual keeps old information too.
8. A feed-forward network processes each word after attention.
9. Standard transformers add learned Q/K/V and multiple heads later.

If you can explain the three-line recipe to someone else, you understand the heart of a transformer. The exercise checks are tiny tests of that understanding."""),
md("""## References

- PyTorch [`Embedding`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html)
- PyTorch [`softmax`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html)
- PyTorch [`Linear`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)

This lesson intentionally teaches single-head identity Q/K/V attention first. It is a stepping stone, not a claim that production transformers omit learned projections or multiple heads."""),
]

for i,cell in enumerate(cells): cell["id"]=f"cell-{i:03d}"
notebook={"cells":cells,"metadata":{"colab":{"name":"A tiny transformer you can see","provenance":[]},"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.10"}},"nbformat":4,"nbformat_minor":5}
OUTPUT.parent.mkdir(exist_ok=True)
OUTPUT.write_text(json.dumps(notebook,indent=1)+"\n",encoding="utf-8")
print(f"wrote {OUTPUT.relative_to(ROOT)} ({len(cells)} cells)")
