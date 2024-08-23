def model_setup(data, mean=[0.25, 5], sd=[0.25, 2.5], generate=False):
    """
    The following function sets up the model.

    Parameters
    ----------
    data : list
        The data as a list of dictionaries.
    mean : list
        The mean of the priors for the parameters.
    sd : list
        The standard deviation of the priors for the parameters.

    Return
    ------
    parameters : cpm.generators.Parameters
        The parameter object for the model
    wrapper : cpm.generators.Wrapper
        The Wrapper object with the model specified below.
    """

    from cpm.models import learning, decision
    from cpm.generators import Value, Parameters, Wrapper
    import numpy as np

    parameters = Parameters(
        # freely varying parameters are indicated by specifying priors
        alpha=Value(
            value=0.5,
            lower=1e-2,
            upper=1,
            prior="truncated_normal",
            args={"mean": mean[0], "sd": sd[0]},
        ),
        beta=Value(
            value=2,
            lower=0,
            upper=10,
            prior="truncated_normal",
            args={"mean": mean[1], "sd": mean[1]},
        ),
        values=np.ones(4) / 4,
    )

    def model_analysis(parameters, trial):
        """
        This function uses the model to estimate latent variables.
        This means that the we use agents choices to estimate the values of the actions.
        """
        # pull out the parameters
        alpha = parameters.alpha
        beta = parameters.beta
        values = np.array(
            parameters.values
        )  # copy essentially prevents us from accidentally overwriting the original values
        # pull out the trial information
        stimulus = np.array(trial.get("trials"))
        feedback = np.array(trial.get("feedback"))
        choice = trial.get("observed")
        choice = choice.astype(int)

        # activate the value of each available action
        # here there are two possible actions, that can take up on 4 different values
        # so we subset the values to only include the ones that are activated...
        # ...according to which stimuli was presented
        activation = values[stimulus - 1]
        # convert the activations to a 2x1 matrix, where rows are actions/outcomes
        activations = activation.reshape(2, 1)
        # calculate a policy based on the activations
        response = decision.Softmax(activations=activations, temperature=beta)
        response.compute()  # compute the policy
        if np.isnan(response.policies).any():
            # if the policy is NaN for a given action, then we need to set it to 1
            print(response.policies)
            response.policies[np.isnan(response.policies)] = 1
            response.policies = response.policies / np.sum(response.policies)
        generated = response.choice()
        # update the value of the chosen action
        mute = np.zeros(4)  # mute learning for all cues not presented
        mute[stimulus[choice] - 1] = 1  # unmute the learning for the chosen action
        reward = feedback[choice]  # get the reward of the chosen action
        teacher = np.array([reward])
        update = learning.SeparableRule(
            weights=values, feedback=teacher, input=mute, alpha=alpha
        )
        update.compute()
        values = values + update.weights  # update the values
        ## compile output
        output = {
            "policy": response.policies,  # policies
            "stimulus": stimulus,  # stimulus presented
            "response": generated,  # choice based on the policy
            "reward": reward,  # reward of the chosen action
            "values": values[0],  # updated values
            "change": update.weights,  # change in the values
            "activation": activations.flatten(),  # activation of the values
            "dependent": np.array([response.policies[1]]),  # dependent variable
        }
        return output

    def model_simulator(parameters, trial):
        """
        This function uses the model generate agents responses.
        This means that we use the values of the actions to generate a choice to simulate real-world data.
        """
        # pull out the parameters
        alpha = parameters.alpha
        beta = parameters.beta
        values = np.array(
            parameters.values
        )  # copy essentially prevents us from accidentally overwriting the original values
        # pull out the trial information
        stimulus = np.array(trial.get("trials"))
        feedback = np.array(trial.get("feedback"))
        choice = trial.get("observed")
        choice = choice.astype(int)

        # activate the value of each available action
        # here there are two possible actions, that can take up on 4 different values
        # so we subset the values to only include the ones that are activated...
        # ...according to which stimuli was presented
        activation = values[stimulus - 1]
        # convert the activations to a 2x1 matrix, where rows are actions/outcomes
        activations = activation.reshape(2, 1)
        # calculate a policy based on the activations
        response = decision.Softmax(activations=activations, temperature=beta)
        response.compute()  # compute the policy
        if np.isnan(response.policies).any():
            # if the policy is NaN for a given action, then we need to set it to 1
            print(response.policies)
            response.policies[np.isnan(response.policies)] = 1
            response.policies = response.policies / np.sum(response.policies)
        generated = response.choice()
        # update the value of the chosen action
        mute = np.zeros(4)  # mute learning for all cues not presented
        mute[stimulus[generated] - 1] = 1  # unmute the learning for the chosen action
        reward = feedback[generated]  # get the reward of the chosen action
        teacher = np.array([reward])
        update = learning.SeparableRule(
            weights=values, feedback=teacher, input=mute, alpha=alpha
        )
        update.compute()
        values = values + update.weights  # update the values
        ## compile output
        output = {
            "policy": response.policies,  # policies
            "stimulus": stimulus,  # stimulus presented
            "response": generated,  # choice based on the policy
            "reward": reward,  # reward of the chosen action
            "values": values[0],  # updated values
            "change": update.weights,  # change in the values
            "activation": activations.flatten(),  # activation of the values
            "dependent": np.array([response.policies[1]]),  # dependent variable
        }
        return output

    if generate:
        model_analysis = model_simulator

    wrapper = Wrapper(model=model_analysis, parameters=parameters, data=data)

    return parameters, wrapper
