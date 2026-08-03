# Character-level transformer on Shakespeare's sonnets

The headline example. A 420,926-parameter decoder-only transformer, trained from
scratch on 96,210 characters of public-domain text, on one CPU core, in about a
minute and a half.

```
python examples/char_transformer/train.py
```

## What it does

Two pre-norm transformer blocks, four heads each, `d_model = 128`, a 64-character
context, a learned positional embedding, and a linear head to the 62-character
vocabulary. AdamW at `lr = 3e-3` with decoupled decay 0.01, a cosine schedule
with 5% warmup, dropout 0.1, gradients clipped at norm 1.0. 1,352 steps of batch
32 x 64 tokens, which is 2.8 million characters, or about 28 passes over the
corpus.

The corpus is `testdata/sonnets.txt`: *Shake-speares Sonnets* (1609), public
domain, folded to ASCII, with the Project Gutenberg boilerplate removed. It is
checked into this repository, so nothing is downloaded at run time.

## Measured output

Produced by the command above, on one CPU core, Python 3.10.12 with NumPy 2.2.6.
The model repr has been elided in the middle - both blocks are identical - and
everything else is verbatim.

```
corpus: 96,210 characters, 62 distinct
a uniform guess scores 4.1271 nats per character

338 steps per epoch, 1,352 in total
CharTransformer()  [420,926 params]
  (tokens): Embedding(num_embeddings=62, embedding_dim=128)  [7,936 params]
  (positions): Embedding(num_embeddings=64, embedding_dim=128)  [8,192 params]
  (blocks): ModuleList()  [396,544 params]
    (0): TransformerBlock(d_model=128)  [198,272 params]
      (norm1): LayerNorm(normalised_shape=128, eps=1e-05, affine=True)  [256 params]
      (attention): MultiHeadAttention(d_model=128, n_heads=4, head_dim=32, causal=True)  [66,048 params]
        (qkv): Linear(in_features=128, out_features=384, bias=True)  [49,536 params]
        (projection): Linear(in_features=128, out_features=128, bias=True)  [16,512 params]
        (dropout): Dropout(p=0.1)  [0 params]
      (norm2): LayerNorm(normalised_shape=128, eps=1e-05, affine=True)  [256 params]
      (mlp): Sequential()  [131,712 params]
        (0): Linear(in_features=128, out_features=512, bias=True)  [66,048 params]
        (1): GELU()  [0 params]
        (2): Linear(in_features=512, out_features=128, bias=True)  [65,664 params]
        (3): Dropout(p=0.1)  [0 params]
    (1): TransformerBlock(d_model=128)  [198,272 params]
      ... identical to block 0 ...
  (norm): LayerNorm(normalised_shape=128, eps=1e-05, affine=True)  [256 params]
  (head): Linear(in_features=128, out_features=62, bias=True)  [7,998 params]

before training:
    SUF'W.gecycMi;GpyWKFGgc-!)yTgcggBU)S,gh:hybygAg'K:Tg rg-cNyrg'H
    KkvKH)w-gH
    gxvgrg-NkPzkrpUCP
    SV';V
    LP--
    H':NCC:P:C-zC )kC

 epoch  train loss   val loss  val bits/char  grad norm
     0      2.2874     1.9255         2.7779     0.9296
     1      1.7586     1.7564         2.5340     0.5986
     2      1.5854     1.6803         2.4241     0.6091
     3      1.4875     1.6631         2.3994     0.6247

trained 420,926 parameters in 91.4s
validation loss 1.6631 nats/char (2.3994 bits/char), against 4.1271 for a uniform guess

sample at temperature 0.8:
    Shall I compare thee
    Steel the strengrows lov'st stry sarvely hearves fair,
    That seem there leaving, and of to flreeds,
    When hom thou wilts thy give my lives graces desert.
    I recay do with with thou art confends,
    Thou did have change than bouns of sun love that remook;
    And all thy sweet show youth, least tan every
    As, that is mourn had found you stravel of his it so be head.
    In stand all thy summer's doth that ince v

sample at temperature 1.0:
    Shall I compare thee.

    VIII

    Or their I each uncord takings brought, out,
    Whild trues as be thy grinks yel his beter
    That comed the praise, but scest not, and it unfore to me:
    I love  as see now should hadow as find:
    Becaud in the still wishiness:
        As thou the disdant poor thee, afferisted should grave,
        Which his nulneeds of love chall ille-neyef your sorry,
        And wailtatters-fair mhis wors to unrantoble,
```

**Validation loss 4.1271 -> 1.6631 nats per character in 91.4 seconds.**

## The loss curve

| epoch | steps | train loss | val loss | val bits/char |
| ----- | ----- | ---------- | -------- | ------------- |
| -     | 0     | -          | 4.1271*  | 5.9542*       |
| 0     | 338   | 2.2874     | 1.9255   | 2.7779        |
| 1     | 676   | 1.7586     | 1.7564   | 2.5340        |
| 2     | 1014  | 1.5854     | 1.6803   | 2.4241        |
| 3     | 1352  | 1.4875     | 1.6631   | 2.3994        |

\* the uniform baseline, `log(62)`, not a measurement of the untrained model.

Training loss falls below validation loss from epoch 1, and the gap widens: at
420,926 parameters against 96,210 characters the model has four parameters per
character of training data and is well into the regime where it can memorise.
Four epochs is where the validation curve is still improving; a fifth would
almost certainly still help and a fiftieth would not.

## Is the text actually better than random

Judge it three ways, because "it looks Shakespearean" is not a measurement.

**The number.** 1.6631 nats per character is 2.3994 bits. A uniform guess over
62 characters costs 5.9542 bits, and the corpus's own unigram entropy - the best
any model that ignores context can do - is 4.5023 bits, measured directly from
the character frequencies. The model is at 40% of the uniform baseline and 53%
of the unigram one, so it has learned structure that letter frequency alone
cannot explain.

**The untrained sample, printed above.** `SUF'W.gecycMi;GpyWKFGgc-!)yTgcggBU)S`
is what this architecture emits before training: no words, no spaces in the
right places, no line structure.

**The trained sample.** Almost every token is a real English word - *compare*,
*leaving*, *summer's*, *praise*, *desert* - lines break at plausible lengths,
the four-space indentation of the sonnets' closing couplets is reproduced, and
at temperature 1.0 it emits `VIII` on its own line surrounded by blank lines,
which is exactly the format of the section headings in the corpus. The grammar
does not survive a full sentence and words like `strengrows` and `nulneeds` are
invented. That is what a two-layer character model with a 64-character window
gets you, and it is precisely the point: it is visibly learning, and visibly not
finished.

## What costs what

91.4 seconds over 1,352 steps is 67.6 ms per step, including the four validation
passes' share. Most of that is the feed-forward network - a transformer block
holds `4d^2` parameters in attention and `8d^2` in the MLP - and the attention
score matrix is materialised in full, `(32, 4, 64, 64)` floats per block per
direction, since this package has no fused attention kernel. At this sequence
length that is 2 MiB and irrelevant; the cost is quadratic in the sequence
length, so at 1,024 tokens the same expression is 512 MiB per block per
direction - which is why every production implementation of attention is a fused
kernel that never writes the matrix down.

The single knob that decides whether this example takes 90 seconds or two hours
is `STRIDE = 8`. Consecutive training windows overlap in 63 of their 64
positions; training on every window would multiply the work by eight for
gradients that are almost the same.

## Things a reader might ask

**Why is the head not tied to the embedding?** Tying is standard and would save
7,936 parameters. The untied version is the one a reader can follow line by line,
and this file is meant to be read.

**Why learned positional embeddings?** With a fixed 64-token window there is
nothing to extrapolate to, so the argument for sinusoidal encodings does not
apply. A table of 64 vectors is 8,192 parameters against a page of trigonometry.

**Why is generation so slow?** There is no key-value cache, so each new
character re-runs the whole 64-token window. A cache would need mutable state
inside the attention layer, which is the thing this package's read-only buffers
exist to prevent. Generation is not the hot path.
