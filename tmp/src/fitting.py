def fitting(wrapper=None, data=None, parallel=True):

    from cpm.optimisation import minimise, FminBound
    from cpm.optimisation.minimise import LogLikelihood

    loss = LogLikelihood.bernoulli

    fit = FminBound(
        model=wrapper,  # Wrapper class with the model we specified from before
        data=data,  # the data as a list of dictionaries
        minimisation=loss,
        prior=False,
        parallel=True,
        cl=None,  # use all available cores
        ppt_identifier="ppt",
        number_of_starts=5,
        approx_grad=True,
        pgtol=1e-10,
    )

    fit.optimise()
