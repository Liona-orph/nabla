# MLP on two moons

A 1,218-parameter MLP separating two interleaving half-circles. The smallest
end-to-end example: dataset, model, training loop, number.

```
python examples/mlp_moons/train.py
```

## What it does

2,000 points, 80/20 split, two `tanh` hidden layers of 32 units, Adam at
`lr = 0.02` for 40 epochs of batch 32. `nabla.data.make_moons` generates the
data from a seed, so this is reproducible on any machine.

Two moons is not linearly separable. A least-squares linear classifier scores
under 92% on it - `tests/test_data.py::test_make_moons_is_not_linearly_separable`
asserts that - so the MLP's 96% is evidence that the hidden layers are doing
something and that the gradients through them are right.

## Measured output

Produced by the command above, on one CPU core, Python 3.10.12 with NumPy 2.2.6.

```
MLP()  [1,218 params]
  (body): Sequential()  [1,152 params]
    (0): Linear(in_features=2, out_features=32, bias=True)  [96 params]
    (1): Tanh()  [0 params]
    (2): Linear(in_features=32, out_features=32, bias=True)  [1,056 params]
    (3): Tanh()  [0 params]
  (head): Linear(in_features=32, out_features=2, bias=True)  [66 params]

 epoch  train loss  train acc  test loss  test acc
     0      0.3290     0.8481     0.2772    0.8700
     5      0.1245     0.9550     0.1117    0.9575
    10      0.0996     0.9625     0.1059    0.9575
    15      0.1037     0.9637     0.0870    0.9650
    20      0.1059     0.9613     0.1114    0.9650
    25      0.1027     0.9650     0.1017    0.9625
    30      0.0969     0.9675     0.0898    0.9700
    35      0.0988     0.9656     0.1033    0.9625
    39      0.1054     0.9619     0.1000    0.9625

trained 1218 parameters for 40 epochs in 0.44s
final test accuracy: 0.9625

decision regions (. = class 0, # = class 1):
................................................
...............................................#
..............................................##
.............................................###
............................................####
..........................................######
.........................................#######
........................................########
.......................................#########
.....................#######..........##########
....................#########........###########
...................###########......############
..................###############.##############
.................###############################
................################################
...............#################################
..............##################################
.............###################################
...........#####################################
..........######################################
.........#######################################
........########################################
.......#########################################
.....###########################################
```

**96.25% test accuracy in 0.44 seconds.**

## Reading the numbers

The test accuracy stops moving after about epoch 15 and the loss stops falling
with it. That is the label noise, not the optimiser: `make_moons` is called with
`noise=0.20`, which puts a few percent of the points on the wrong side of the
true boundary, and no classifier can recover those. Training accuracy and test
accuracy stay within half a percent of each other for the whole run, so the
model is not overfitting - it has hit the ceiling of the problem.

The decision-region plot shows the interlocking crescents: the strip of `.`
running down the middle-right and the wedge of `#` reaching up into the
upper-left of it are the two moons' tails. A linear classifier's plot would be a
single straight diagonal.

## Things a reader might ask

**Why two output logits for a two-class problem?** One logit with
`binary_cross_entropy` is exactly equivalent and one parameter smaller. Using
`cross_entropy` here means all three examples share a loss function, which is
worth more than the parameter.

**Why `tanh` rather than ReLU?** The true boundary is curved and a ReLU
network's is piecewise linear. At 32 units the corners are visible in the error
rate; `tanh` costs one transcendental per unit and gives a smooth boundary.
