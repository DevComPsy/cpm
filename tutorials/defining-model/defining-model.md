# Build your model

In `cpm` the way you build models is by writing a function that specifies the transformation from your independent variables to your dependent variables for a single trial.
This means that the code you need to write is rather negligible.
It makes it easy to focus on the things that matter the most - specifying the model!

## The function

Let us quickly write some function that specifies our model. On each trial, it will have to know the stimuli that were presented and the feedback that it gets for each action. It will also need to know the parameters that it can use to make decisions.

For now, let us assume that we have a trial that looks like this:

```python
## create a single trial as a dictionary
trial = {
    "trials": np.array([1, 2]),
    "feedback": np.array([1, 0]),
}
```

Here we hav all the information we need for a given trial - all input to the model other than the parameters and its initial state. In reinforcement learning, this is what we call the state of the environment. The model will use this information to make a decision and update its internal state.


One advantage of `cpm` is that most components that you will need to build sequential decision-making models are already implemented. This means that you can focus on the model itself, rather than the implementation details. Here we will use the `learning` and `decision` modules to build a simple model based on the Rescorla-Wagner update rule and a Greedy-decision rule.

```python
from cpm.models import learning, decision, utils
import copy

def model(parameters, trial):
    # pull out the parameters
    alpha = parameters.alpha
    temperature = parameters.temperature
    values = np.array(parameters.values)
    # pull out the trial information
    stimulus = trial.get('trials')
    feedback = trial.get("feedback")
    mute = np.zeros(4)  # mute learning for all cues not presented

    # activate the value of each available action
    # here there are two possible actions, that can take up on 4 different values
    # so we subset the values to only include the ones that are activated...
    # ...according to which stimuli was presented
    activation = values[stimulus - 1]
    # convert the activations to a 2x1 matrix, where rows are actions/outcomes
    activations = activation.reshape(2, 1)
    # calculate a policy based on the activations
    response = decision.Softmax(activations=activations, temperature=epsilon)
    response.compute() # compute the policy
    choice = response.choice() # get the choice based on the policy
    reward = feedback[choice] # get the reward of the chosen action

    
    # update the value of the chosen action
    mute[stimulus[choice] - 1] = 1 # unmute the learning for the chosen action
    teacher = np.array([reward])
    update = learning.SeparableRule(weights=values, feedback=teacher, input=mute, alpha=alpha)
    update.compute()
    values += update.weights.flatten()
    ## compile output
    output = {
        "policy"   : response.policies,         # policies
        "response" : choice,                    # choice based on the policy
        "reward"   : reward,                    # reward of the chosen action
        "values"   : values,                    # updated values
        "change"   : update.weights,            # change in the values
        "activation" : activations.flatten(),     # activation of the values
        "dependent"  : response.policies,        # dependent variable
    }
    return output
```

If you want to learn more about this model, you can check out the [tutorial on the model](../examples/example2.ipynb).

The immediately obvious thing is that **the function takes two arguments**:

* **parameters** : the freely-varying parameters of the model and its initial state of the model. It must be a [`cpm.Parameters`](../references/generators.md#cpm.generators.Parameters) class. We already covered it in the [tutorial on parameters](parameters.md).
* **trial** : this essentially includes all the information that we will need to do the computations we specified in the model. This is pulled from the data we covered in the [data format](data-format.md) section.

The function should return a dictionary that includes all the information that you want to save from the model. This can include the dependent variables, the policy, the response, the reward, the values, and the change in the values. You can also include any other information that you might need for analysis. For example, if you want to update the `values` in the `parameters` object, you can simply include it in the output, and it will update it in the parameters. Similarly if you want to know any variables on a trial-by-trial level that is not part of the `parameters` object, `cpm` will save it as long as it is part of the model function output.

## Applying it to data

So, where do you loop through the trials? `cpm` has built-in tools that frees you up from writing complicated for loops to apply the model to each trial. `cpm` also compiles the data you need into a neat `pandas.DataFrame`. We will use the [`cpm.generators.Wrapper`](../references/generators.md#cpm.generators.Wrapper) for this. Wrapper only does the simulation for one participant, so the data we need to input is a dictionary as opposed to a list of dictionaries.

```python
from cpm.generators import Wrapper
decision_model = Wrapper(model = model, parameters = parameters, data = data)
```

If you want to run the model on the data, simply use the `run()` method, after which you can `export()` the simulation details as a `pandas.DataFrame`:

```python
decision_model.run()
decision_model_output = decision_model.export()
```

## Simulating participants

Now we usually have many participants, each with different trial order and distribution of rewards. What we can do here is to use the [`cpm.generators.Simulator`](../references/generators.md#cpm.generators.Simulator) to apply the model to all participants' trial orders.

```python
from cpm.generators import Simulator
simulator = Simulator(wrapper = decision_model, parameters = parameters, data = complete_data)
simulator.run()
simulation = simulator.export()
```

This works largely as `Wrapper` does, apart from two caveats:

* The data we provide for `Simulator` is a list of dictionaries, see [data format](data-format.md) section.
* The parameters required for the simulation can be a Parameters object or a list of dictionaries whose length is equal to data. If it is a Parameters object, Simulator will use the same parameters for all simulations. It is a list of dictionaries, it will use match the parameters with data, so that for example parameters[6] will be used for the simulation of data[6].
