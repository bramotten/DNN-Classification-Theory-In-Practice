Try `main.ipynb`; it also works on Colab.

The goal of this project is to test the influence of the small value bound index α and Hölder smoothness index β on neural network classification performance. Performance refers to conditional probability approximation rather than most probable class prediction. Slightly more on the "small value bound" follows below.

We sample (multi-dimensional) features from some specified distribution. This "distribution" can also be just a density function; in this case we rejection sample from it. Anyway, these features serve as input for some specified "conditional class probability" functions, one for every output class. The outputs are normalized if necessary, resulting in a conditional probability vector per sample. That serves as probability vector of a categorical distribution from which the final label/class of a sample is drawn. 

The networks are trained for minimal categorical cross-entropy / negative log likelihood with the features and the one-hot encoded label (which may not be the most likely one!) -- but we will see the conditional probability functions back later. The network implementation is fairly straightforward Keras/TensorFlow, though we want to be able to easily modify some network parameters like the number and widths of layers and L1 penalty. As is common anyway but essential here, ReLU activation functions are used throughout while the final output goes through a softmax. 

We are primarily interested in the _convergence rate_ of the Küllback-Leibler divergence between the true conditional class probability distribution and the one predicted by the DNN. Again, training did _not_ use the true probabilities -- only the one-hot encoding sampled from a categorical distribution.

The KL divergence is interesting because the non-constant aspect of a theoretical bound you can place on it features [given caveats] an `n ** (-(1 + α) * β / ((1 + α) * β + d))` _or_ `n ** (-2 * β / (2 * β + d))` term, where `n` is the sample size, `d` the number of features, and `β` the Hölder smoothness-index. The proportion of small probabilities, which can be measured by the small value bound with index `α`, is less well known and detailed in the thesis or [see next paragraph]. 

[The theoretical results are here](https://arxiv.org/abs/2108.00969) and are somewhat similar to  [this work for nonparametric regression by one of the supervisors of this project](https://arxiv.org/abs/1708.06633) and [this work that considers 0/1-error risk in classification](https://arxiv.org/abs/1812.03599).
