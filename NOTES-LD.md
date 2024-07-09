# notes on variational bayes implementation

I will keep track of my thoughts while going through the variational bayes implementation here.
It is well written and modular, so it is important to go through it as we will make use of the general structure in the future for complex methods.

As I went along, I also made some changes to the comments - I will not list theme here. I made them feel less like GPT text :-)
I also changed some comments into docstrings. Generally, class methods should have docstrings, given that they are part of the public API - unless they are private, which should be indicated with a leading underscore.

Also, it would be a good idea to reference exact equations, so that we can track the implementation with the theory better. I did do it when I could, but I will try to do it more often. The same goes for empirical bayes and individual components, which is on my to-do list for a while.
For now, I added some TODO items where references were missing and I thought that you might now where they are coming from.

Line 66: self.hyperprios is a dictionary, so it does not provide a way to access the hyperpriors for a specific parameter with self.hyperpriors.v, you need to use self.hyperpriors.get('v'). This is tedious and error-prone, so I will change the hyperpriors to be class Parameters with attributes instead of a dictionary.
Line 70: Added some warnings to the code to let users know that they did not define the hyperpriors - remind them of good and open practices :)

Line 479: Convergence is always tracked via parameters, changed it to be a user-defined option with deafault=True.

# Questions

## Convergence Tracking

We should implement the same method to track convergence across both hierarchical methods.
In EmpiricalBayes, we essentially stop when the objective loss function plateaus.
The way it is done here is pretty good, but we could potentially do some improvements - we could use Cohen's _d_ or something similar. I will think about it and let you know.

Also, normalised mean might be a bit too flexible - we could potentially use a more strict convergence criterion.
Right now, we have this:
mean = 1, sd = 0.5, normalised-mean=2
mean = 0.5, sd = 0.25, normalised-mean=2
mean = 0.6, sd = 0.3, normalised-mean=2
mean = 0.7, sd = 0.35, normalised-mean=2
Therefore, algorithm has converged.

Some possible solutions that come to mind are Kullback-Leibler divergence, Jensen-Shannon divergence, or effect size measures.
I will think about it.

## one-sample t-test

With a one sample t-test, we should have more summary stats in the output.
We should probably include the mean, standard deviation, and the t-statistic, confidence intervals, as well as the p-value and the null hypothesis we tested against.
We might also just throw in an effect size - although people could do that themselves.

Response from Frank:
The output dataframe `t_df` builds on the `self.hyperparameters` dataframe, so it already includes - for each parameter - the estimated population-level mean, standard error of that mean, and the estimated population-level standard deviation. We then add the t-statistics and the p-values.
We could indeed add confidence intervals and effect sizes. We know the exact form of the marginal posterior over the population-level mean (Piray et al. Equation 24, page 30). So we could provide a 95% quantile interval of that distribution as the confidence interval.
The effect size a la Cohen's d is directly proportional to the t-statistic, so I'm not sure how much added value that would have.

## Adding Output

Since we can determine the exact posterior distributions over the participant-level parameters and the population-level means and standard deviations, we could add more details in the output.
In terms of participant-level parameters, we currently only return the MAP parameter estimates - which correspond to the posterior means - but we could also return the participant-level standard deviations (obtained from Hessian matrices), and even 95% quantile intervals of the normal posterior distributions (based on the estimated means and SDs).
In terms of the population-level means and SDs of parameters, we could similarly return their posterior means, SDs and 95% QIs.