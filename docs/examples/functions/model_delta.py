def model_delta(data):
    import numpy
    import cpm
    import pandas
    
    parameters = cpm.generators.Parameters(
        # free parameters are indicated by specifying priors
        alpha=cpm.generators.Value(
            value=0.5,
            lower=1e-10,
            upper=1,
            prior="truncated_normal",
            args={"mean": 0.5, "sd": 0.25},
        ),
        temperature=cpm.generators.Value(
            value=1,
            lower=0,
            upper=10,
            prior="truncated_normal",
            args={"mean": 5, "sd": 2.5},
        ),
        # everything without a prior is part of the initial state of the
        # model or constructs fixed throughout the simulation
        # (e.g. exemplars in general-context models of categorizations)
        # initial q-values starting starting from non-zero value
        # these are equal to all 4 stimuli (1 / 4)
        values = numpy.array([0.25, 0.25, 0.25, 0.25])
        )

    def model_generator(parameters, trial):
        # pull out the parameters
        alpha = parameters.alpha
        temperature = parameters.temperature
        values = numpy.asarray(parameters.values)
        
        # pull out the trial information
        stimulus = numpy.array([trial.arm_left, trial.arm_right]).astype(int)
        feedback = numpy.array([trial.reward_left, trial.reward_right])

        # Equation 1. - get the value of each available action
        # Note that because python counts from 0, we need to shift
        # the stimulus identifiers by -1
        expected_rewards = values[stimulus - 1]
        # convert columns to rows
        expected_rewards = expected_rewards.reshape(2, 1)
        # calculate a policy based on the activations
        # Equation 2.
        choice_rule = cpm.models.decision.Softmax(
            activations=expected_rewards,
            temperature=temperature
            )
        choice_rule.compute() # compute the policy
        # if the policy is NaN for an action, then we need to set it to 1
        # this corrects some numerical issues with python and infinities
        if numpy.isnan(choice_rule.policies).any():
            choice_rule.policies[numpy.isnan(choice_rule.policies)] = 1
        model_choices = choice_rule.choice()  # get the model's choice
        # get the received reward for the choice
        reward = feedback[model_choices]
        teacher = numpy.array([reward])
        # we now create a vector that tells our learning rule what...
        # ... stimulus to update according to the participant's choice
        what_to_update = numpy.zeros(4)
        chosen_stimulus = stimulus[model_choices] - 1
        what_to_update[chosen_stimulus] = 1

        # Equation 4.
        update = cpm.models.learning.SeparableRule(
                        weights=values,
                        feedback=teacher,
                        input=what_to_update,
                        alpha=alpha
                        )
        update.compute()
        # Equation 5.
        values += update.weights.flatten()
        # compile output
        output = {
            "trial"    : trial.trial.astype(int), # trial numbers
            "activation" : expected_rewards.flatten(), # expected reward of arms
            "policy"   : choice_rule.policies,       # policies
            "reward"   : reward,                  # received reward
            "error"    : update.weights,          # prediction error
            "values"   : values,                  # updated values
            "response" : model_choices,          # model's choice
            # dependent variable
            "dependent"  : numpy.array([choice_rule.policies[1]]),
        }
        return output

    def model_fitting(parameters, trial):
        # pull out the parameters
        alpha = parameters.alpha
        temperature = parameters.temperature
        values = numpy.asarray(parameters.values)
        
        # pull out the trial information
        stimulus = numpy.array([trial.arm_left, trial.arm_right]).astype(int)
        feedback = numpy.array([trial.reward_left, trial.reward_right])
        human_choice = trial.observed.astype(int)

        # Equation 1. - get the value of each available action
        # Note that because python counts from 0, we need to shift
        # the stimulus identifiers by -1
        expected_rewards = values[stimulus - 1]
        # convert columns to rows
        expected_rewards = expected_rewards.reshape(2, 1)
        # calculate a policy based on the activations
        # Equation 2.
        choice_rule = cpm.models.decision.Softmax(
            activations=expected_rewards,
            temperature=temperature
            )
        choice_rule.compute() # compute the policy
        # if the policy is NaN for an action, then we need to set it to 1
        # this corrects some numerical issues with python and infinities
        if numpy.isnan(choice_rule.policies).any():
            choice_rule.policies[numpy.isnan(choice_rule.policies)] = 1
        # get the received reward for the choice
        reward = feedback[human_choice]
        teacher = numpy.array([reward])
        # we now create a vector that tells our learning rule what...
        # ... stimulus to update according to the participant's choice
        what_to_update = numpy.zeros(4)
        chosen_stimulus = stimulus[human_choice] - 1
        what_to_update[chosen_stimulus] = 1

        # Equation 4.
        update = cpm.models.learning.SeparableRule(
                        weights=values,
                        feedback=teacher,
                        input=what_to_update,
                        alpha=alpha
                        )
        update.compute()
        # Equation 5.
        values += update.weights.flatten()
        # compile output
        output = {
            "trial"    : trial.trial.astype(int), # trial numbers
            "activation" : expected_rewards.flatten(), # expected reward of arms
            "policy"   : choice_rule.policies,       # policies
            "reward"   : reward,                  # received reward
            "error"    : update.weights,          # prediction error
            "values"   : values,                  # updated values
            # dependent variable
            "dependent"  : numpy.array([choice_rule.policies[1]]),
        }
        return output

    # values = numpy.asarray(parameters.values)
    model_one_generative = cpm.generators.Wrapper(
        model=model_generator,
        parameters=parameters,
        data=data[data.ppt == 1],
    )

    model_one_fitting = cpm.generators.Wrapper(
        model=model_fitting,
        parameters=parameters,
        data=data[data.ppt == 1],
    )

    return model_one_generative, model_one_fitting, parameters