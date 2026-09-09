"""Build only Notebook 04: a beginner next-word DNN."""
import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks/04_first_dnn_next_word.ipynb"

def md(s):
    return {"cell_type":"markdown","metadata":{},"source":dedent(s).strip()+"\n"}

def code(s, tags=None):
    return {"cell_type":"code","execution_count":None,"metadata":{"tags":tags} if tags else {},"outputs":[],"source":dedent(s).strip()+"\n"}

cells = [
md("""# Your first text neural network: finish the chant!

We will teach one tiny **dense neural network (DNN)** to predict the next word:

```text
DUCK DUCK DUCK → GOOSE!
TIC TAC         → TOE!
HIP HIP         → HOORAY!
READY SET       → GO!
```

The model reads exactly three words. It uses token embeddings, positional
embeddings, one hidden layer, and one output score for every possible next word.
There is no attention and no downloaded dataset. Everything is small enough to
draw, print, and understand."""),
md("""## The whole model

```text
3 words → 3 token vectors ─┐
                           + → flatten → dense layer → ReLU → next-word scores
positions → 3 position vec ┘

DUCK DUCK DUCK                                      highest score: GOOSE
```

Training changes numbers inside embeddings and dense layers until correct next
words receive high scores."""),
code("""import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import torch
from torch import nn
from torch.nn import functional as F

SEED=24
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.set_printoptions(precision=3,sci_mode=False)
print("PyTorch",torch.__version__)"""),
md("""## 1. Beginner ML vocabulary

- **Example:** one question/answer pair: `DUCK DUCK DUCK → GOOSE`.
- **Input/features:** the three context words.
- **Label/target:** the correct next word.
- **Parameter:** a number the model can change.
- **Prediction:** the model's answer.
- **Loss:** one number measuring wrongness; lower is better.
- **Training:** predict, measure loss, adjust parameters, repeat.
- **Epoch:** one learning pass over all examples.
- **Accuracy:** fraction of answers that are correct.

This model memorizes tiny patterns. It does not understand general English—and that
is fine. A small task makes every moving part visible."""),
md("""## 2. Our entire dataset

Each chant repeats in a circle. Sliding a three-word window over
`DUCK DUCK DUCK GOOSE DUCK ...` makes:

```text
DUCK DUCK DUCK  → GOOSE
DUCK DUCK GOOSE → DUCK
DUCK GOOSE DUCK → DUCK
GOOSE DUCK DUCK → DUCK
```"""),
code("""patterns=[
    ["DUCK","DUCK","DUCK","GOOSE"],
    ["TIC","TAC","TOE"],
    ["HIP","HIP","HOORAY"],
    ["RED","LIGHT","GREEN","LIGHT"],
    ["READY","SET","GO"],
    ["ONE","TWO","THREE","GO"],
]
context_size=3
examples=[]
for pattern in patterns:
    stream=pattern*20
    for i in range(len(stream)-context_size):
        examples.append((stream[i:i+context_size],stream[i+context_size]))
print("Examples:",len(examples))
for context,target in examples[:8]: print(" "," ".join(context),"→",target)"""),
code("""fig,ax=plt.subplots(figsize=(9,2.2)); ax.axis("off")
for i,word in enumerate(["DUCK","DUCK","DUCK","GOOSE"]):
    color="#bde0fe" if i<3 else "#ffafcc"
    box=FancyBboxPatch((i*1.7,.7),1.35,.65,boxstyle="round,pad=.08",fc=color,ec="#333")
    ax.add_patch(box); ax.text(i*1.7+.675,1.025,word,ha="center",va="center",weight="bold")
ax.text(1.7,.35,"INPUT: three words",ha="center"); ax.text(5.8,.35,"TARGET",ha="center")
ax.add_patch(FancyArrowPatch((4.85,1.02),(5.15,1.02),arrowstyle="->",mutation_scale=18))
ax.set_xlim(-.2,7); ax.set_ylim(0,1.8); plt.title("One training example"); plt.show()"""),
md("""### Python/R bridge

Python starts indexing at 0. `stream[i:i+3]` returns exactly three values because
the upper endpoint is excluded. `for context, target in examples` unpacks each
two-part example into two names."""),
md("""### Your turn 1 — make the vocabulary

Make a sorted list containing every unique word once.

**Exact API help**

- `set(values)` removes duplicates, like R's `unique()`.
- `sorted(values)` sorts.
- Put all words together with `[w for pattern in patterns for w in pattern]`."""),
code("vocabulary = None  # TODO",["exercise"]),
code("""if vocabulary is None: print("🟡 Use sorted(set(...)).")
else:
    assert vocabulary==sorted(set(w for p in patterns for w in p))
    assert len(vocabulary)==16
    print("🟢",len(vocabulary),"words:",vocabulary)""",["check"]),
code("""# Reference values keep later sections runnable before exercises are finished.
_vocabulary=sorted(set(w for pattern in patterns for w in pattern))
word_to_id={word:i for i,word in enumerate(_vocabulary)}
id_to_word={i:word for word,i in word_to_id.items()}
vocab_size=len(_vocabulary)
print(word_to_id)"""),
code("""counts={w:0 for w in _vocabulary}
for _,target in examples: counts[target]+=1
plt.figure(figsize=(10,3.5)); plt.bar(counts.keys(),counts.values(),color="#669bbc")
plt.xticks(rotation=45,ha="right"); plt.ylabel("times used as target")
plt.title("The 16 possible next words"); plt.tight_layout(); plt.show()"""),
md("""## 3. Turn words into integer tensors

PyTorch uses rectangular numeric arrays called **tensors**. Inputs have shape
`(examples,3)` and targets have shape `(examples,)`. Use `torch.long` integers;
embedding tables and classification targets require integer IDs.

### Your turn 2 — encode the data

**Exact API help**

- `word_to_id[word]` gives one ID.
- `torch.tensor(values,dtype=torch.long)` makes an integer tensor.
- Inputs: `[[word_to_id[w] for w in context] for context,target in examples]`.
- Targets: `[word_to_id[target] for context,target in examples]`."""),
code("""input_rows = None
target_values = None
X_ids = None
y_ids = None""",["exercise"]),
code("""if X_ids is None: print("🟡 Build the two lists, then wrap them in torch.tensor.")
else:
    assert X_ids.shape==(len(examples),3) and y_ids.shape==(len(examples),)
    assert X_ids.dtype==y_ids.dtype==torch.long
    print("🟢",X_ids.shape,y_ids.shape)""",["check"]),
code("""_X=torch.tensor([[word_to_id[w] for w in context] for context,target in examples],dtype=torch.long)
_y=torch.tensor([word_to_id[target] for context,target in examples],dtype=torch.long)
print("inputs",_X.shape,"targets",_y.shape)"""),
md("""## 4. Token embedding table

IDs are arbitrary labels. An embedding table gives every word a learnable vector.
We use two numbers per word so we can plot them:

```text
16 vocabulary words × 2 numbers = table shape (16,2)
```

The table is not square. Vocabulary size and embedding size are unrelated.

### Your turn 3 — create token embeddings

**Exact API help**

- `nn.Embedding(num_embeddings,embedding_dim)` creates the table.
- Use `nn.Embedding(vocab_size,2)`.
- Calling `token_embedding(_X)` retrieves rows."""),
code("""embedding_size=2
token_embedding = None
token_vectors = None""",["exercise"]),
code("""if token_embedding is None: print("🟡 token_embedding=nn.Embedding(vocab_size,embedding_size)")
else:
    assert token_embedding.weight.shape==(16,2)
    assert token_vectors.shape==(len(examples),3,2)
    print("🟢 IDs",_X.shape,"→ vectors",token_vectors.shape)""",["check"]),
md("""## 5. Positional embeddings

Position 0 is the oldest word; position 2 is newest. Add a learned position vector
to each token vector. The network then knows both **what word** and **where**.

### Your turn 4 — add positions

**Exact API help**

- `nn.Embedding(context_size,embedding_size)` creates `(3,2)`.
- `torch.arange(context_size)` creates IDs `[0,1,2]`.
- Call the table on IDs, then add the result to token vectors."""),
code("""_temporary_tokens=nn.Embedding(vocab_size,embedding_size)(_X)
position_embedding = None
position_ids = None
position_vectors = None
combined_vectors = None""",["exercise"]),
code("""if position_embedding is None: print("🟡 Start with nn.Embedding(context_size,embedding_size).")
else:
    assert position_embedding.weight.shape==(3,2)
    assert combined_vectors.shape==(len(examples),3,2)
    print("🟢 token + position:",combined_vectors.shape)""",["check"]),
md("""## 6. Flatten and build the DNN

Three words × two numbers becomes six numbers: `(N,3,2) → (N,6)`. Flattening only
rearranges values.

The network is:

```text
6 inputs → 16 hidden neurons → ReLU → 16 next-word logits
```

`nn.Linear` connects every input to every output. `ReLU` changes negative values to
zero, allowing nonlinear patterns. Logits are scores, not probabilities."""),
code("""fig,ax=plt.subplots(figsize=(11,3)); ax.axis("off")
labels=[("3 word IDs","#bde0fe"),("token + position\\nembeddings","#a2d2ff"),("flatten\\n6 numbers","#cdb4db"),("dense + ReLU\\n16 neurons","#ffc8dd"),("16 next-word\\nlogits","#ffafcc")]
xs=[.2,2.3,4.5,6.5,9]
for (label,color),x in zip(labels,xs):
    box=FancyBboxPatch((x,.9),1.55,.9,boxstyle="round,pad=.08",fc=color,ec="#333")
    ax.add_patch(box); ax.text(x+.775,1.35,label,ha="center",va="center")
for a,b in zip(xs[:-1],xs[1:]): ax.add_patch(FancyArrowPatch((a+1.57,1.35),(b-.03,1.35),arrowstyle="->",mutation_scale=18))
ax.set_xlim(0,10.8); ax.set_ylim(.5,2.2); plt.title("Our complete next-word DNN"); plt.show()"""),
md("""### Your turn 5 — create the DNN

**Exact API help**

- `nn.Linear(in_features,out_features)` makes a dense layer.
- `nn.ReLU()` takes no arguments.
- `nn.Sequential(layer1,layer2,...)` chains layers.
- Use `Linear(6,16)`, `ReLU()`, `Linear(16,vocab_size)`."""),
code("dnn = None  # TODO",["exercise"]),
code("""if dnn is None: print("🟡 Use nn.Sequential with the three listed layers.")
else:
    assert dnn(torch.zeros(2,6)).shape==(2,16)
    print("🟢 two examples → 16 logits each")""",["check"]),
md("""## 7. Put every piece into one model

`nn.Module` is PyTorch's model base class. `__init__` creates layers; `forward`
describes the path from input IDs to logits. Calling `model(inputs)` runs `forward`."""),
code("""class ChantDNN(nn.Module):
    def __init__(self,vocab_size,context_size=3,embedding_size=2,hidden_size=16):
        super().__init__()
        self.context_size=context_size
        self.token_embedding=nn.Embedding(vocab_size,embedding_size)
        self.position_embedding=nn.Embedding(context_size,embedding_size)
        self.network=nn.Sequential(
            nn.Linear(context_size*embedding_size,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,vocab_size))
    def forward(self,context_ids):
        positions=torch.arange(self.context_size,device=context_ids.device)
        vectors=self.token_embedding(context_ids)+self.position_embedding(positions)
        return self.network(vectors.flatten(start_dim=1))

model=ChantDNN(vocab_size)
print(model)
print("Trainable parameters:",sum(p.numel() for p in model.parameters()))"""),
md("""## 8. Before training: random guesses

Softmax changes logits into probabilities adding to 1. `argmax` finds the largest.
Before training, parameters and predictions are random."""),
code("""def encode(words): return torch.tensor([[word_to_id[w] for w in words]])
probe=encode(["DUCK","DUCK","DUCK"])
with torch.no_grad(): before_probs=F.softmax(model(probe)[0],dim=0)
plt.figure(figsize=(10,3.5)); plt.bar(_vocabulary,before_probs.numpy(),color="#adb5bd")
plt.xticks(rotation=45,ha="right"); plt.ylabel("probability"); plt.ylim(0,1)
plt.title("Before training: DUCK DUCK DUCK → ?"); plt.tight_layout(); plt.show()
print("Random answer:",id_to_word[before_probs.argmax().item()])"""),
md("""## 9. Loss measures wrongness

Cross-entropy compares logits with target IDs. Pass raw logits—not probabilities.
Random guessing among 16 words has loss near `ln(16) = 2.77`. Lower is better.

### Your turn 6 — calculate loss

**Exact API help**

- `model(_X)` gives logits shaped `(N,16)`.
- `F.cross_entropy(logits,_y)` gives one scalar loss."""),
code("""all_logits = None
first_loss = None""",["exercise"]),
code("""if first_loss is None: print("🟡 all_logits=model(_X); first_loss=F.cross_entropy(all_logits,_y)")
else:
    assert all_logits.shape==(len(examples),16) and first_loss.ndim==0
    print("🟢 initial loss:",first_loss.item())""",["check"]),
md("""## 10. Training loop

One epoch is five lines:

```python
logits = model(inputs)                 # guess
loss = F.cross_entropy(logits,targets) # measure
optimizer.zero_grad()                 # erase old gradients
loss.backward()                       # calculate directions
optimizer.step()                      # update parameters
```

Create the optimizer with `torch.optim.Adam(model.parameters(),lr=0.03)`. The
learning rate is the update step size."""),
md("""# Answer key and complete payoff

This independent model runs even if exercises are unfinished. Compare it with your
work, then watch it learn."""),
code("""answer_vocab=sorted(set(w for pattern in patterns for w in pattern))
answer_stoi={w:i for i,w in enumerate(answer_vocab)}
answer_itos={i:w for w,i in answer_stoi.items()}
answer_X=torch.tensor([[answer_stoi[w] for w in context] for context,target in examples])
answer_y=torch.tensor([answer_stoi[target] for context,target in examples])
answer_model=ChantDNN(len(answer_vocab))
initial_embeddings=answer_model.token_embedding.weight.detach().clone()
optimizer=torch.optim.Adam(answer_model.parameters(),lr=.03)
print("inputs",answer_X.shape,"labels",answer_y.shape)"""),
code("""losses=[]; accuracies=[]; goose_probability={}
for epoch in range(201):
    logits=answer_model(answer_X)
    loss=F.cross_entropy(logits,answer_y)
    accuracy=(logits.argmax(1)==answer_y).float().mean()
    if epoch in [0,5,20,200]:
        duck=torch.tensor([[answer_stoi["DUCK"]]*3])
        goose_probability[epoch]=F.softmax(answer_model(duck)[0],dim=0)[answer_stoi["GOOSE"]].item()
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    losses.append(loss.item()); accuracies.append(accuracy.item())

fig,axes=plt.subplots(1,2,figsize=(11,3.5))
axes[0].plot(losses,color="#c1121f"); axes[0].set(xlabel="epoch",ylabel="loss",title="Wrongness falls")
axes[1].plot(accuracies,color="#2a9d8f"); axes[1].set(xlabel="epoch",ylabel="accuracy",title="Correct answers rise",ylim=(0,1.05))
plt.tight_layout(); plt.show()
print(f"loss {losses[0]:.3f} → {losses[-1]:.5f}")
print(f"accuracy {accuracies[0]:.1%} → {accuracies[-1]:.1%}")
assert accuracies[-1]>.99"""),
code("""plt.figure(figsize=(6,3.5)); plt.plot(goose_probability.keys(),goose_probability.values(),marker="o",linewidth=3)
for epoch,p in goose_probability.items(): plt.text(epoch,p+.04,f"{p:.0%}",ha="center")
plt.ylim(0,1.12); plt.xlabel("epoch"); plt.ylabel("P(GOOSE)")
plt.title("DUCK DUCK DUCK → GOOSE becomes believable"); plt.grid(alpha=.2); plt.show()"""),
code("""learned=answer_model.token_embedding.weight.detach()
fig,axes=plt.subplots(1,2,figsize=(13,5))
for ax,points,title in zip(axes,[initial_embeddings,learned],["Before: random","After: useful for chants"]):
    ax.axhline(0,color="lightgray"); ax.axvline(0,color="lightgray")
    for i,w in enumerate(answer_vocab):
        ax.scatter(*points[i],s=55); ax.annotate(w,points[i],xytext=(4,4),textcoords="offset points",fontsize=9)
    ax.set(xlabel="embedding feature 1",ylabel="embedding feature 2",title=title); ax.grid(alpha=.15)
plt.tight_layout(); plt.show()"""),
code("""def answer_encode(words): return torch.tensor([[answer_stoi[w] for w in words]])
with torch.no_grad(): probs=F.softmax(answer_model(answer_encode(["DUCK"]*3))[0],dim=0)
colors=["#ffafcc" if w=="GOOSE" else "#a2d2ff" for w in answer_vocab]
plt.figure(figsize=(10,3.5)); plt.bar(answer_vocab,probs.numpy(),color=colors)
plt.xticks(rotation=45,ha="right"); plt.ylabel("probability"); plt.ylim(0,1.05)
plt.title("After training: DUCK DUCK DUCK → GOOSE!"); plt.tight_layout(); plt.show()
print("Prediction:",answer_itos[probs.argmax().item()],f"({probs.max().item():.1%})")"""),
md("""## 11. Payoff: continue the chant

Generation repeatedly reads the latest three words, predicts one, appends it, and
repeats."""),
code("""@torch.no_grad()
def continue_chant(start_words,new_words=10):
    output=list(start_words)
    for _ in range(new_words):
        logits=answer_model(answer_encode(output[-3:]))[0]
        output.append(answer_itos[logits.argmax().item()])
    return " ".join(output)

for prompt in [["DUCK","DUCK","DUCK"],["TIC","TAC","TOE"],["HIP","HIP","HOORAY"],["READY","SET","GO"]]:
    print(continue_chant(prompt))"""),
code("""# Change these to any three vocabulary words.
YOUR_PROMPT=["ONE","TWO","THREE"]
print(continue_chant(YOUR_PROMPT,12))"""),
md("""## 12. Learned positional embeddings

One 2D vector is learned for each context position."""),
code("""points=answer_model.position_embedding.weight.detach()
plt.figure(figsize=(5,5)); plt.axhline(0,color="lightgray"); plt.axvline(0,color="lightgray")
for i,p in enumerate(points):
    plt.arrow(0,0,p[0],p[1],head_width=.08,length_includes_head=True)
    plt.text(p[0],p[1],f" position {i}")
plt.xlabel("position feature 1"); plt.ylabel("position feature 2")
plt.title("Three learned position vectors"); plt.grid(alpha=.2); plt.show()"""),
md("""## What it learned—and did not learn

It memorized our chants. That is also **overfitting**: an unseen combination may
produce confident nonsense. A serious project separates training, validation, and
test data. Exact duplicates make such a split misleading here, so we honestly call
this a memorization demo.

The ingredients are real: embeddings, positions, logits, softmax, cross-entropy,
gradients, and Adam all appear in much larger text models."""),
md("""## Exercise answers

```python
vocabulary=sorted(set(w for pattern in patterns for w in pattern))
input_rows=[[word_to_id[w] for w in context] for context,target in examples]
target_values=[word_to_id[target] for context,target in examples]
X_ids=torch.tensor(input_rows,dtype=torch.long)
y_ids=torch.tensor(target_values,dtype=torch.long)
token_embedding=nn.Embedding(vocab_size,2)
token_vectors=token_embedding(_X)
position_embedding=nn.Embedding(3,2)
position_ids=torch.arange(3)
position_vectors=position_embedding(position_ids)
combined_vectors=_temporary_tokens+position_vectors
dnn=nn.Sequential(nn.Linear(6,16),nn.ReLU(),nn.Linear(16,vocab_size))
all_logits=model(_X)
first_loss=F.cross_entropy(all_logits,_y)
```"""),
md("""## Debugging checklist and references

- Inputs `(N,3)`; targets `(N,)`; both `torch.long`.
- Embeddings `(N,3,2)`; flattened `(N,6)`; output `(N,16)`.
- Pass logits—not probabilities—to cross-entropy.
- `zero_grad()` before `backward()`; `step()` afterward.
- Print shapes whenever confused.

APIs: [`torch.tensor`](https://docs.pytorch.org/docs/stable/generated/torch.tensor.html),
[`nn.Embedding`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html),
[`nn.Linear`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html),
[`nn.ReLU`](https://docs.pytorch.org/docs/stable/generated/torch.nn.ReLU.html),
[`cross_entropy`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html),
[`Adam`](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html).

You should not need these docs to finish: every required argument appears directly
before its exercise."""),
]

for i,c in enumerate(cells): c["id"]=f"cell-{i:03d}"
nb={"cells":cells,"metadata":{"colab":{"name":"Your first DNN: finish the chant","provenance":[]},"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.10"}},"nbformat":4,"nbformat_minor":5}
OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps(nb,indent=1)+"\n",encoding="utf-8")
print(f"wrote {OUT.relative_to(ROOT)} ({len(cells)} cells)")
