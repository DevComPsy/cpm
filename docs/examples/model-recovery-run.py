import cpm
import warnings
from packaging import version


if __name__ == "__main__":
    import numpy
    import pandas as pd
    import cpm
    import cpm.datasets as datasets
    import functions as f
    
    data = datasets.load_bandit_data()
    data["observed"] = data["response"].astype(int)  # convert responses to int
    data.head()

    model_one_generative, model_one_fitting, parameters = f.model_delta(data)
    model_two_generative, model_two_fitting, parameters_two = f.model_kernel(data)
    model_three_generative, model_three_fitting, parameters_three = f.model_anticorrelated(data)
    # values = numpy.asarray(parameters.values)
    generators = (
        model_one_generative,
        model_two_generative,
        model_three_generative,
    )

    parameter_sets = (
        parameters,
        parameters_two,
        parameters_three,
    )

    fitting = (
        model_one_fitting,
        model_two_fitting,
        model_three_fitting,
    )

    model_names = (
        "delta",
        "kernel",
        "anticorrelated"
    )

    dataset = data.copy()


    bigout = pd.DataFrame()
    for x in numpy.arange(100):
        print(f"Run {x + 1} of 100")
        for i in numpy.arange(3):
            simulator = cpm.generators.Simulator(
                wrapper=generators[i],
                parameters=parameter_sets[i].sample(dataset.ppt.nunique()),
                data=data.groupby("ppt")
            )
            simulator.run()
            out = simulator.export()
            dataset["observed"] = out.response_0
            for m in numpy.arange(3):
                model = fitting[m]
                ## mute runtime warings
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                fit = cpm.optimisation.FminBound(
                    model=model,  # Wrapper class with the model we specified from before
                    data=dataset.groupby('ppt'),  # the data as a list of dictionaries
                    minimisation=cpm.optimisation.minimise.LogLikelihood.bernoulli,
                    parallel=True,
                    prior=False,
                    ppt_identifier="ppt",
                    display=False,
                    number_of_starts=5,
                    # everything below is optional and passed directly to the scipy implementation of the optimiser
                    approx_grad=True
                )
                fit.optimise()
                output = fit.export()
                output["target"] = model_names[i]
                output["model"] = model_names[m]
                output["run"] = x
                bigout = pd.concat([bigout, output], ignore_index=True)

    bigout.to_csv("./solutions/05-model-recovery-results.csv", index=False)
