# Specify your parameters

In `cpm`, we specify parameters via the `Parameters` class. This class works similar to the `dict` class, and so behaves like a dictionary, but with several extended functionalities. This means that you can access the parameters using the `[]` operator, and you can also use the `update` method to update the parameters.
The `Parameters` class also includes the initial state of the model, which `cpm` separates from freely varying parameters in the model by requiring users to define priors for the freely varying parameters but not for the initial state. Most models will have a single initial state that do not require priors.

In terms of the actual code, look below:

```python
from cpm.generators import Value, Parameters
import numpy as np

parameters = Parameters(
    # freely varying parameters are indicated by specifying priors
    alpha=Value(
        value=0.5,
        lower=1e-10,
        upper=1,
        prior="truncated_normal",
        args={"mean": 0.5, "sd": 0.25},
    ),
    temperature=Value(
        value=1,
        lower=1e-10,
        upper=10,
        prior="truncated_normal",
        args={"mean": 5, "sd": 2.5},
    ),
    values=np.array([[0.25, 0.25, 0.25, 0.25]]),
)
```

In the above code snippet, we define a `Parameters` object called `parameters`. We specify two freely varying parameters, `alpha` and `temperature`, and their lower and upper bounds, the prior distribution, and the arguments for the prior distribution. We also specify the initial state of the model, `values`, which is a 2D numpy array. The `Value` class is used to specify the starting Q-value vector, or a similar variable.

Some of the more important functionalities of the `Parameters` class are sampling parameters given the prior and getting the prior probability of the parameters given the prior.

## Sampling

The sampling procedure is fairly straightforward with the `Parameters.sample()` method:

```python
parameters.sample(5)
```

This would generate random deviates of the parameter from their prior distributions. The output would look something like this:

```python
[
    {"alpha": 0.7627162320187792, "temperature": 0.7115049580842472},
    {"alpha": 0.39805853951904824, "temperature": 9.10024224906725},
    {"alpha": 0.4662189894139982, "temperature": 3.150982328012355},
    {"alpha": 0.05421332309089305, "temperature": 3.234912688658496},
    {"alpha": 0.3707025149977554, "temperature": 6.240271855060641},
]
```

## Calculating the probability of your parameters

Calculating the prior probability is quite simple as well.

```python
params.PDF()
# It would produce: 0.07771262020833226
```

If you are new to priors, one thing that you will come across is that this number is not between 0 and 1. This is because the prior probability is not a probability in the traditional sense. It is a density, and so it can be greater than 1. This is because discrete and continuous variables are not defined in the same way. In the likelihood function that we use in model fitting when we have discrete responses, will indeed be a probability between 0 and 1. This is not true for continuous variables, so don't be surprised. If you wish to learn more, there are some great threads on StackOverflow you can have a look to learn about PDF, probabilities, and likelihoods:

* Newman (https://math.stackexchange.com/users/325579/newman), How can a probability density function (pdf) be greater than 1?, URL (version: 2016-03-30): [https://math.stackexchange.com/q/1720053](https://math.stackexchange.com/q/1720053)
* John Doe (https://stats.stackexchange.com/users/12223/john-doe), What is the reason that a likelihood function is not a pdf?, URL (version: 2012-07-06): [https://stats.stackexchange.com/q/31238](https://stats.stackexchange.com/q/31238)



