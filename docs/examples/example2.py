# %% [markdown]
# # Example 2: Reinforcement learning with a two-armed bandit.
#
# This is an **intermediate** level tutorial, where we assume that you are familiar with the basics of reinforcement learning and the two-armed bandit problem.
# In this example, we will apply and fit a reinforcement learning model to a two-armed bandit problem.
# The model will be consist of a $\epsilon$-greedy policy and a prediction error term.

# %% [markdown]
# ## Multi-armed bandits
#
# Multi-armed bandit problems are a class of reinforcement learning problems where the person is faced with multiple choices, each with a different degree of reward.
# The goal of the person is to learn which choice is the best and to maximize the reward over time.
# In this example, we will consider a two-armed bandit problem, where the person is faced with two choices (select an item on the left or the item on the right), each with a different reward.
# There are 4 different items that can appear in combinations of two, and the reward for each item varies.
# For example, if item 1 has a chance of 0.7 of giving a reward, then you can expect to receive a reward 70% of the time when you select item 1.
# The problem is that the underlying structure of the item-reward mapping is unknown.
# Here we will use the following item-reward mapping:
#
# | Item | Reward |
# |------|--------|
# | 1    | 0.8    |
# | 2    | 0.2    |
# | 3    | 0.5    |
# | 4    | 0.9    |

# %% [markdown]
# ## Import the data
#
# First, we will import the data and get it ready for the toolbox.

# %%
import pandas as pd
import numpy as np

experiment = pd.read_csv("bandit.csv", header=0)
experiment.head(10)

# %% [markdown]
# Let us look at what each column represents:
#
# - `left`: the stimulus presented on the left side.
# - `right`: the stimulus presented on the right side.
# - `reward_left`: the reward received when the left stimulus is selected.
# - `reward_right`: the reward received when the right stimulus is selected.
# - `ppt`: the participant number.
#
# Notice that for now, we don't have any actual data recorded from participants.
# That is because, for now, we will only import the environment, containing all states and rewards.
# So, let us convert the data into a format that the toolbox can understand.

# %% [markdown]
# Fortunately, the toolbox provides a function to convert the data into the required format.
# We will use the `pandas_to_dict` function available in the `cpm.utils` module.
#
#

# %%
from cpm.utils import pandas_to_dict

experiment = pandas_to_dict(
    experiment, participant="ppt", stimuli="stim", feedback="reward"
)
length = len(experiment)
print(f"Number of participants: {length}")

# %% [markdown]
# Here, we have a list of dictionaries, where each dictionary represents an experimental session that a participant might complete.
# If you have 100 participants or sessions, then you will have 100 dictionaries in the list -each with their unique trial order (schedule).

# %%
print(f"Key variables in the dictionary: {experiment[0].keys()}")
print(
    f"Number of trials and number of the stimuli on each of those trials: {experiment[0].get('trials').shape}"
)
print(
    f"Number of trials and number of the feedback on each of those trials: {experiment[0].get('feedback').shape}"
)
print("All looks good! We are ready to go!")


# %% [markdown]
#
# Let us see what each value within the dictionary represents:
#
# - `input`: the stimuli presented in the session. It must bve a numpy.ndarray. Each row represents a trial, and the columns represent the left and right stimuli presented in each trial.
# - `feedback`: the rewards that could be obtained from the stimuli. It must be a numpy.ndarray. Each row represents a trial, and the columns represent the reward that could be obtained from the corresponding stimulus in the `stimuli` key.

# %% [markdown]
# ## The model
#
# Let us quickly go through the model we will use.
#
# Each stimulus has an associated value, which is the expected reward that can be obtained from selecting that stimulus.
#
# Let $Q(a)$ be the estimated value of action $a$.
# On each trial, $t$, there are two stimuli present, so that $Q(a)$ could be $Q(\text{left})$ or $Q(\text{right})$, where the corresponding Q-values are derived from the associated value of the stimuli present on left or right.
#
# On each trial $t$, the $\epsilon$-greedy policy selects a random action with probability $\epsilon$, the exploration rate parameter, and selects the action with the highest estimated value with probability $1-n\epsilon$, where $n$ the number of possible actions.
# So, on each trial, the model will select an action (left or right) based on the following policy:
#
# $$
# A_t =
# \begin{cases}
# \text{random action} & \text{with probability } \epsilon \\
# \arg\max_a Q_t(a) & \text{with probability } 1 - n \epsilon
# \end{cases}
# $$
#
# where $A_t$ is the action selected at time $t$, and $Q_t(a)$ is the estimated value of action $a$ at time $t$.
#
# The model will update the estimated value of the selected action using the following learning rule:
#
# $$
# \Delta Q_t(A_t) = \alpha \times \Big[ R_t - Q_t(A_t) \Big]
# $$
#
# where $\alpha$ is the learning rate, and $R_t$ is the reward received at time $t$.
# Q-values are then updated as follows:
#
# $$
# Q_{t+1}(A_t) = Q_t(A_t) + \Delta Q_t(A_t)
# $$
#
#
#
# ### Building the model
#
# In order to use the toolbox, you will have to specify **the computations for one single trial**.
# Rest assured, you do not have to build the model from scratch.
# We have fully-fledged models in `cpm.applications` that you can use, but we also have all the building blocks in `cpm.components` that you can use to build your own model.
#
# For now, let us simplify the problem and start by specifying what information we need on each trial.
# This information will usually be extracted from the data we just imported.
# Here, we create this to help us develop the model.
#
# Here, we need to specify the following:
#
# - `stimuli`: the stimuli presented in the trial.
# - `rewards`: the rewards that could be obtained from selecting the stimuli.

# %%
trial = {"input": [1, 4], "feedback": [1, 0]}

# %% [markdown]
# Now, before we build the model, we also have to talk about the `cpm.Parameter` class.
# This class is used to specify the parameters of the model, including various bounds and priors.
# Let us specify the parameters for the model.

# %%
from cpm.models import Parameters

parameters = Parameters(alpha=0.12, epsilon=0.1, values=np.array([0.25, 0.25, 0.25, 0.25]))
print(f"The learning rate of the model: {parameters.alpha.export()}")
print(f"The exploration rate of the model: {parameters.epsilon.export()}")
print(f"The initial value of each action: {parameters.values.export()}")

# %% [markdown]
# You can immediately see that the toolbox defined a bunch of things for us.
# This includes the prior and the parameter ranges as well.
# One thing we have to clarify is the value for each action.
# Here we initialized the value of each action to 0.
# This is somewhat special here, because it is a 2D array.
# The reason is because each stimuli is actionable, and each stimuli is a single unit.
# If we had compound stimulus on each side, varying on various dimensions (color, shape, etc.), then we would have multiple columns instead of just one.
# One such example is Niv et al. (2015), where they had three compound stimuli on the screen, that varied on dimensions of color, shape, and fill type.
# The way you represent it will also depend on the model building blocks you use and how you specify the computations for each trial.
# Nonetheless, even though we treated it as a parameter, we do not need to estimate it later on.
#
# That was enough preparation, let us build a model.

# %%
from cpm.components import learning, decision, utils


def model(parameters, trial):
    # pull out the parameters
    alpha = parameters.alpha
    epsilon = parameters.epsilon.value
    values = parameters.values.value.copy()
    # pull out the trial information
    stimulus = np.array(trial.get("input")).flatten()
    feedback = trial.get("feedback")
    teacher = np.zeros(4)  # teaching signal for the learning term
    mute = np.zeros(4)  # mute learning for all cues not presented

    # activate the value of each available action
    # here there are two possible actions, that can take up on 4 different values
    # so we subset the values to only include the ones that are activated...
    # ...according to which stimuli was presented
    activation = values[stimulus - 1]
    # convert the activations to a 2x1 matrix, where rows are actions/outcomes
    activations = activation.reshape(2, 1)
    # calculate a policy based on the activations
    response = decision.GreedyRule(activations=activations, epsilon=epsilon)
    response.compute()  # compute the policy
    choice = response.choice()  # make a choice
    reward = feedback[choice]  # get the reward of the chosen action

    # update the value of the chosen action
    teacher[stimulus[choice] - 1] = (
        reward  # update the teacher's value of the chosen action
    )
    mute[stimulus[choice] - 1] = 1  # unmute the learning for the chosen action
    update = learning.SeparableRule(
        weights=values, feedback=teacher, input=mute, alpha=alpha
    )
    update.compute()
    values += update.weights.flatten()
    ## compile output
    output = {
        "policy": response.policies,  # policies
        "response": choice,  # choice based on the policy
        "reward": reward,  # reward of the chosen action
        "values": values,  # updated values
        "change": update.weights,  # change in the values
        "activation": activations.flatten(),  # activation of the values
    }
    return output


model(parameters, trial)

# %% [markdown]
# The important bit here is that values, response and policy must be returned by the function you specify.
# They will be used by other methods in the toolbox to indentify key variables.

# %% [markdown]
# ### Simulate data

# %%
# from cpm.models import Simulator, Wrapper

wrapper = Wrapper(model=model, parameters=parameters, data=experiment[0])
wrapper.run()

# %% [markdown]
# ## Parameter Recovery
