"""A character-level transformer trained on Shakespeare's sonnets.

This is the example that has to work. Everything else in the package -
broadcasting, the attention mask, the stable cross-entropy, AdamW's decoupled
decay, the cosine schedule, gradient clipping - is exercised here at once, and
the output is a paragraph you can read and judge for yourself.

# The model

A 420,000-parameter decoder: a character embedding, a *learned* positional
embedding, two pre-norm transformer blocks with four heads each, a final
`LayerNorm`, and a linear head to the 62-character vocabulary.

Learned positions rather than sinusoidal, because with a fixed 64-token window
there is nothing to extrapolate to, and a table of 64 vectors is 8,192
parameters against a page of trigonometry.

The output head is *not* tied to the input embedding. Tying them is standard and
would save 7,936 parameters, and this file leaves them separate because the
untied version is the one a reader can follow line by line.

# What it costs

About 90 seconds on one CPU core for 1,352 steps of batch 32 x 64 tokens, which
is 2.8 million characters, or about 28 passes over the corpus. Every matrix
product is a single-threaded NumPy call; see `benchmarks/` for what that costs
against a hand-written baseline.

Run it with:

    python examples/char_transformer/train.py
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

import nabla as nb
from nabla import data, nn, optim, train

SEED = 20240601
CORPUS = Path(__file__).resolve().parents[2] / "testdata" / "sonnets.txt"

CONTEXT = 64
D_MODEL = 128
HEADS = 4
LAYERS = 2
DROPOUT = 0.1

BATCH_SIZE = 32
EPOCHS = 4
LEARNING_RATE = 3e-3
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0

#: One training example per `STRIDE` characters rather than per character.
#: Consecutive windows overlap in 63 of their 64 positions, so training on all
#: of them spends 64 times the arithmetic to see almost the same gradients; at
#: stride 8 an epoch still covers every character in eight different positions.
#: This is the single knob that decides whether the example takes 90 seconds or
#: two hours.
STRIDE = 8

#: Validation is sampled more coarsely still. It only has to be an unbiased
#: estimate, and 150 windows of 64 characters is 9,600 predictions.
VALIDATION_STRIDE = 64


class CharTransformer(nn.Module):
    """Embedding, positions, blocks, norm, head.

    The forward pass is six lines, and every one of them is an operation whose
    gradient `check_grad` has verified against a central difference. That is the
    entire claim this package makes, and this is where it is cashed in.
    """

    def __init__(
        self,
        vocabulary_size: int,
        generator: nb.random.Generator,
        *,
        context: int = CONTEXT,
        d_model: int = D_MODEL,
        heads: int = HEADS,
        layers: int = LAYERS,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.context = context
        self.tokens = self.register_module(
            "tokens", nn.Embedding(vocabulary_size, d_model, generator=generator)
        )
        self.positions = self.register_module(
            "positions", nn.Embedding(context, d_model, generator=generator)
        )
        self.blocks = self.register_module(
            "blocks",
            nn.ModuleList(
                nn.TransformerBlock(
                    d_model,
                    heads,
                    generator=generator,
                    max_length=context,
                    dropout=dropout,
                )
                for _ in range(layers)
            ),
        )
        # The final normalisation is not optional in a pre-norm stack: each
        # block adds to the residual stream and nothing on that path ever
        # rescales it, so the input to the head would otherwise grow with depth.
        self.norm = self.register_module("norm", nn.LayerNorm(d_model))
        self.head = self.register_module(
            "head", nn.Linear(d_model, vocabulary_size, generator=generator)
        )

    def forward(self, ids: nb.Tensor | np.ndarray) -> nb.Tensor:
        """Map `(batch, time)` character ids to `(batch, time, vocabulary)` logits."""
        length = ids.shape[-1]
        if length > self.context:
            msg = f"sequence of {length} exceeds the context of {self.context}"
            raise nb.ShapeError(msg)
        hidden = self.tokens(ids) + self.positions(np.arange(length))
        for block in self.blocks:
            hidden = block(hidden)
        return self.head(self.norm(hidden))


def generate(
    model: CharTransformer,
    vocabulary: data.CharVocabulary,
    prompt: str,
    *,
    length: int,
    temperature: float,
    generator: nb.random.Generator,
) -> str:
    """Sample `length` characters, one at a time, continuing `prompt`.

    Quadratic in `length` and deliberately so: there is no key-value cache, so
    every new character re-runs the whole window. A cache would make this
    linear and would need mutable state inside the attention layer, which is
    exactly the kind of thing this package's read-only buffers are there to
    prevent. Generation is not the hot path; training is.

    Temperature divides the logits before the softmax. Below 1 the distribution
    sharpens towards the argmax - fewer spelling mistakes, more repetition -
    and above 1 it flattens.
    """
    if temperature <= 0:
        msg = f"temperature must be positive, got {temperature}"
        raise ValueError(msg)
    rng = generator.numpy_generator
    context = list(vocabulary.encode(prompt))
    model.eval()
    with nb.no_grad():
        for _ in range(length):
            window = np.array(context[-model.context :], dtype=np.int64)[None, :]
            logits = model(window).numpy()[0, -1] / temperature
            probabilities = np.exp(logits - logits.max())
            probabilities /= probabilities.sum()
            context.append(int(rng.choice(len(probabilities), p=probabilities)))
    return vocabulary.decode(context)


def main() -> None:
    """Train the model, report the loss curve, and print two samples."""
    text = CORPUS.read_text(encoding="ascii")
    vocabulary = data.CharVocabulary(text)
    tokens = vocabulary.encode(text)
    uniform = math.log(vocabulary.size)
    print(f"corpus: {len(text):,} characters, {vocabulary.size} distinct")
    print(f"a uniform guess scores {uniform:.4f} nats per character")
    print()

    # Split by position, not at random. Random windows would put a window
    # starting at character 100 in the training set and one starting at 101 in
    # the validation set, and 63 of their 64 characters are the same: the
    # validation loss would then be measuring memorisation.
    cut = int(0.9 * len(tokens))
    train_windows = data.NextCharDataset(tokens[:cut], length=CONTEXT)
    validation_windows = data.NextCharDataset(tokens[cut:], length=CONTEXT)

    generator = nb.random.Generator(seed=SEED)
    loader_generator, model_generator, sample_generator = generator.spawn(3)

    train_loader = data.DataLoader(
        data.Subset(train_windows, np.arange(0, len(train_windows), STRIDE)),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=loader_generator,
        drop_last=True,
    )
    validation_loader = data.DataLoader(
        data.Subset(validation_windows, np.arange(0, len(validation_windows), VALIDATION_STRIDE)),
        batch_size=BATCH_SIZE,
    )
    steps = EPOCHS * len(train_loader)
    print(f"{len(train_loader):,} steps per epoch, {steps:,} in total")

    model = CharTransformer(vocabulary.size, model_generator)
    print(model)
    print()

    trainer = train.Trainer(
        model,
        optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY),
        nn.cross_entropy,
        schedule=optim.schedule.cosine_with_warmup(
            LEARNING_RATE, warmup_steps=steps // 20, total_steps=steps
        ),
        grad_clip=GRAD_CLIP,
    )

    print("before training:")
    print(
        _quote(
            generate(
                model, vocabulary, "S", length=120, temperature=0.8, generator=sample_generator
            )
        )
    )
    print()

    started = time.perf_counter()
    history = trainer.fit(train_loader, epochs=EPOCHS, val_loader=validation_loader, verbose=False)
    elapsed = time.perf_counter() - started
    model.train()

    print(
        f"{'epoch':>6}  {'train loss':>10}  {'val loss':>9}  "
        f"{'val bits/char':>13}  {'grad norm':>9}"
    )
    for record in history.epochs:
        print(
            f"{int(record['epoch']):>6}  {record['train_loss']:>10.4f}  "
            f"{record['val_loss']:>9.4f}  {record['val_loss'] / math.log(2):>13.4f}  "
            f"{record['train_grad_norm']:>9.4f}"
        )

    final = history.last()
    print()
    print(f"trained {model.parameter_count():,} parameters in {elapsed:.1f}s")
    print(
        f"validation loss {final['val_loss']:.4f} nats/char "
        f"({final['val_loss'] / math.log(2):.4f} bits/char), "
        f"against {uniform:.4f} for a uniform guess"
    )
    print()

    for temperature in (0.8, 1.0):
        print(f"sample at temperature {temperature}:")
        print(
            _quote(
                generate(
                    model,
                    vocabulary,
                    "Shall I compare thee",
                    length=400,
                    temperature=temperature,
                    generator=sample_generator,
                )
            )
        )
        print()


def _quote(text: str) -> str:
    """Indent a block so it is visibly the model's output and not the program's."""
    return "\n".join("    " + line for line in text.splitlines())


if __name__ == "__main__":
    main()
