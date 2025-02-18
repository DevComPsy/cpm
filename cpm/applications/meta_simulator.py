import numpy as np
import pandas as pd
import copy
import pickle as pkl

from cpm.core.optimisers import LinearConstraint
from cpm.optimisation.minimise import LogLikelihood

from ..generators.parameters import Parameters
from ..core.data import unpack_participants
from ..core.generators import cast_parameters
from ..core.exports import simulation_export
from ..optimisation import FminBound


class MetaSignalDetectionSimulator:

    def __init__(
        self,
        wrapper=None,
        data=None,
        parameters=None,
        solver="fminbound",
        parallel=False,
        ppt_identifier="ppt",
        num_trials=None,
    ):
        self.model = wrapper
        self.data = data
        assert solver == "fminbound", "Only fminbound is supported for now."
        self.solver = solver
        self.__parallel__ = parallel
        self.ppt_identifier = ppt_identifier

        self.groups = None
        self.__run__ = False
        self.__pandas__ = isinstance(data, pd.api.typing.DataFrameGroupBy)
        self.__parameter__pandas__ = isinstance(parameters, pd.DataFrame)
        self.__parameter__dict__ = isinstance(parameters, dict)
        if isinstance(self.__pandas__, pd.DataFrame):
            raise TypeError(
                "Data should be a pandas.DataFrameGroupBy object, not a pandas.DataFrame."
            )
        if self.__pandas__:
            self.groups = list(self.data.groups.keys())
        else:
            self.groups = np.arange(len(self.data))

        self.parameters = cast_parameters(parameters, len(self.groups))
        self.parameter_names = self.model.parameter_names

        self.num_trials = num_trials

        self.simulation = {}
        self.generated = {}

        m = copy.deepcopy(self.model)
        m.reset(
            parameters=self.parameters[self.groups[0]],
            data=self.data,
            ppt=self.groups[0],
        )
        m.run()
        self.nbins = m.simulation.get("nbins")

    def predict(self):
        for i, ppt in enumerate(self.groups):
            self.model.reset()
            evaluate = copy.deepcopy(self.model)
            ppt_data = unpack_participants(
                self.data, i, self.groups, pandas=self.__pandas__
            )
            ppt_parameter = (
                unpack_participants(
                    self.parameters, i, self.groups, pandas=self.__parameter__pandas__
                )
                if not self.__parameter__dict__
                else self.parameters[ppt]
            )
            evaluate.reset(parameters=ppt_parameter, data=ppt_data)
            evaluate.run()
            output = copy.deepcopy(evaluate.simulation)
            self.simulation[ppt] = output.copy()
            del evaluate, output

        self.simulation = np.array(self.simulation, dtype=object)
        self.__run__ = True
        return None

    def sample(self, num_trials=None, num_samples=100):

        nt = {}
        if num_trials is None:
            for i, ppt in enumerate(self.groups):
                ppt_data = unpack_participants(
                    self.data, i, self.groups, pandas=self.__pandas__
                )
                nt[ppt] = len(ppt_data)
        else:
            for ppt in self.groups:
                nt[ppt] = num_trials

        evaluate = copy.deepcopy(self.model)
        evaluate.reset(
            parameters=self.parameters[self.groups[0]],
            data=self.data,
            ppt=self.groups[0],
        )
        evaluate.run()

        nR_S1 = np.zeros((len(self.groups), num_samples, 2 * self.nbins))
        nR_S2 = np.zeros((len(self.groups), num_samples, 2 * self.nbins))

        for idx, (ppt, nt) in enumerate(nt.items()):
            self.model.reset()
            evaluate = copy.deepcopy(self.model)
            ppt_data = unpack_participants(
                self.data, idx, self.groups, pandas=self.__pandas__
            )
            ppt_parameter = (
                unpack_participants(
                    self.parameters, idx, self.groups, pandas=self.__parameter__pandas__
                )
                if not self.__parameter__dict__
                else self.parameters[ppt]
            )
            evaluate.reset(parameters=ppt_parameter, data=ppt_data)
            nR_S1[idx], nR_S2[idx] = evaluate.sample_ppt(
                num_trials=nt, num_samples=num_samples, ppt=ppt
            )

        generated = []
        for ppt_idx, ppt in enumerate(self.groups):
            for i in range(num_samples):
                observed = [
                    np.array(
                        [
                            nR_S1[ppt_idx, i, : self.nbins],
                            nR_S2[ppt_idx, i, : self.nbins],
                            nR_S2[ppt_idx, i, : self.nbins],
                            nR_S1[ppt_idx, i, : self.nbins],
                        ]
                    )
                ]
                generated.append(
                    {
                        "ppt": (ppt, i),
                        "nbins": self.nbins,
                        "observed": observed,
                    }
                )

        self.generated = pd.DataFrame(generated)

        return None

    def run(self, verbose=True, num_starts=5, num_samples=100):

        self.num_samples = num_samples

        self.predict()
        self.sample(num_samples=num_samples, num_trials=self.num_trials)

        data_pandas = self.generated

        model = copy.deepcopy(self.model)
        model.reset(
            parameters=self.parameters[self.groups[0]],
            data=self.data,
            ppt=self.groups[0],
        )
        model.run()
        A = np.zeros((2 * self.nbins - 1, 2 * self.nbins - 1))
        for i in range(2 * self.nbins - 3):
            A[i, i + 1 : i + 3] = [-1, 1]
        A[-2, self.nbins - 1] = -1
        A[-1, self.nbins] = 1
        b = np.zeros(2 * self.nbins - 1)
        constraints = LinearConstraint(A=A, b=b)

        fit = FminBound(
            model=self.model,
            data=data_pandas,
            minimisation=LogLikelihood.multinomial_float,
            ppt_identifier=self.ppt_identifier,
            parallel=self.__parallel__,
            prior=False,
            display=False,
            number_of_starts=num_starts,
            # everything below is optional and passed directly to the scipy implementation of the optimiser
            approx_grad=True,
            constraints=constraints,
            verbose=verbose,
        )

        fit.optimise()
        self.results = fit.export()
        return None

    def export(self):

        results = self.results.copy()
        results["ppt"] = np.array(
            [self.num_samples * [ppt] for ppt in self.groups]
        ).flatten()

        xs = [col for col in results.columns if col.startswith("x_")]
        result = results.groupby("ppt")[xs].agg(["mean", "std"]).reset_index()

        names = ["meta_d1"] + [f"t2c1_{i}" for i in range(self.nbins - 2)]
        result.rename(
            columns={f"x_{i}": names[i] for i in range(len(names))}, inplace=True
        )

        return result
