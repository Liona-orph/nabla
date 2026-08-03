# Ten-class classifier on generated digits

A 16,522-parameter MLP on 8x8 digit-like images. Nothing is downloaded and
nothing is checked in: `nabla.data.make_digits` renders ten hand-drawn glyphs
and jitters them, deterministically, from a seed.

```
python examples/mnist_style/train.py
```

## What it does

6,000 training images and 2,000 test images, each a glyph translated by up to
two pixels in each direction, scaled to a random intensity between 0.6 and 1.0,
and corrupted with Gaussian noise of standard deviation 0.25. The model is
`64 -> 96 -> 96 -> 10` with ReLU, Kaiming initialisation, dropout at 0.2, AdamW
with a cosine schedule and 5% warmup, and gradient clipping at 1.0.

There is no convolution in this package, so the images are flattened. The
translation invariance a convolution would give for free has to be learned,
which is exactly what makes the shifted digits hard for an MLP.

## Measured output

Produced by the command above, on one CPU core, Python 3.10.12 with NumPy 2.2.6.

```
three training examples:
 :    ..      = ---   @*   +@ 
- @@@#@    %@**%%#-   @#...@@ 
.@%.  #*   @@   +@+   -#+@*+@ 
:@@ = *#       .%=+    .   #@.
. @+@##     :. @+ :   .:  @% :
 @@   @@      *@-      =-+#   
.@@   @@     -%%- .     =  -= 
  @@@%@=    :.@# :-    : .  : 
labels: [8, 7, 9]

Sequential()  [16,522 params]
  (0): Linear(in_features=64, out_features=96, bias=True)  [6,240 params]
  (1): ReLU()  [0 params]
  (2): Dropout(p=0.2)  [0 params]
  (3): Linear(in_features=96, out_features=96, bias=True)  [9,312 params]
  (4): ReLU()  [0 params]
  (5): Dropout(p=0.2)  [0 params]
  (6): Linear(in_features=96, out_features=10, bias=True)  [970 params]

 epoch  train loss  train acc  test loss  test acc
     0      2.3454     0.1320     2.0861    0.3300
     5      0.4610     0.8412     0.2572    0.9245
    10      0.2611     0.9055     0.1555    0.9455
    15      0.1973     0.9308     0.1277    0.9520
    20      0.1606     0.9417     0.1129    0.9545
    25      0.1479     0.9450     0.1118    0.9540
    29      0.1467     0.9488     0.1112    0.9560

trained 16522 parameters for 30 epochs in 1.2s
final test accuracy: 0.9560

confusion matrix (rows true, columns predicted):
         0    1    2    3    4    5    6    7    8    9
   0   202    0    0    0    0    0    2    0    0    1
   1     0  177    0    0    4    0    0    0    0    0
   2     0    0  197    0    0    0    0    1    0    0
   3     0    0    0  163    2    2    0    0   15    1
   4     0    1    0    1  194    0    0    0    0    0
   5     1    0    0    1    0  217    0    2    0    0
   6     0    0    1    0    0    0  220    0    5    1
   7     2    0    0    0    1    0    0  197    0    2
   8     0    0    0   17    0    0    8    0  166    4
   9     0    0    0    3    3    0    5    0    2  179
```

**95.60% test accuracy in 1.2 seconds.**

## Reading the numbers

The confusion matrix is where the interesting information is, and it is not
uniform. 32 of the 88 errors are the pair 3/8 in both directions, and another 13
are 6/8. Those glyphs differ by two lit pixels on the left edge, and at a
translation of two pixels plus noise of 0.25 that difference is close to the
noise floor. The remaining classes are essentially solved.

The test accuracy is *higher* than the training accuracy throughout, which looks
wrong and is not: dropout is active during training and disabled during
evaluation, so the training number is the accuracy of a randomly thinned network
and the test number is the accuracy of the whole one.

## Does the dropout earn its place

Measured, by running the same script with `DROPOUT = 0.0` and with `DROPOUT = 0.2`
and changing nothing else:

| dropout | train accuracy | test accuracy |
| ------- | -------------- | ------------- |
| 0.0     | 0.9993         | 0.9525        |
| 0.2     | 0.9488         | 0.9560        |

Half a point of test accuracy, which on 2,000 examples is seven images and is
not a result anyone should lean on. The generalisation gap is the real story:
4.7 points without dropout, and negative with it. The undropped model has
memorised the training set - 16,522 parameters against 6,000 examples - and the
dropped one has not.

## Things a reader might ask

**Why 8x8 and not 28x28?** The point of the example is that the training stack
works on a ten-class problem, and a 28x28 input would multiply the runtime by
twelve to demonstrate the same thing.

**Why is this not MNIST?** Because MNIST would have to be downloaded, and an
example that reaches the network is an example that fails on an aeroplane. The
cost is stated plainly: this is a jittered bitmap font, not handwriting, and
95.6% here says nothing about 95.6% on real digits.
