from audioop import mul
import numpy as np
from scipy.stats import multivariate_normal
from ..generators import Parameters, Value
from ..models import MetaSignalDetectionHelper, MetaSignalDetectionModel



def metacognitionSDT_initparams(data = None, binned_data = None, nbins = 4, s = 1):

    assert data is not None or binned_data is not None, "Either data or binned_data must be provided."

    helper = MetaSignalDetectionHelper(data = None, nbins = 4)
    if data is None:
        helper.nR_S1 = np.array(binned_data['nR_S1'])
        helper.nR_S2 = np.array(binned_data['nR_S2'])
        helper.d1 = helper.compute_d1()
        helper.c1 = helper.compute_c1()
        helper.t1c1 = helper.c1[helper.nbins-1]
        helper.t2c1 = helper.c1 - helper.t1c1
        helper.t2c1 = np.delete(helper.t2c1, helper.nbins-1)
        helper.data_pandas = helper.create_df()

    params = Parameters(
        meta_d1 = Value(
            value = helper.d1,
            lower = 0,
            prior = "truncate_normal",
            args = {"mean": 0, "sd": 5},
        ),
        t2c1 = Value(
            value = helper.t2c1,
            lower = np.where(np.linspace(-2, 2, 2*nbins-2) < 0, -100, 0),
            upper = np.where(np.linspace(-2, 2, 2*nbins-2) > 0, 100, 0),
            prior = multivariate_normal,
            args = {"mean": np.delete(np.linspace(-2, 2, 2*nbins-1), nbins), "cov": np.eye(2*nbins-2)},
        ),
        t1c1 = helper.t1c1,
        s = s,
        d1 = helper.d1,
    )

    return params, helper.d1, helper.t1c1, helper.data_pandas


def metacognitionSDT_model(parameters):

    meta_d1 = parameters.meta_d1
    meta_c1 = parameters.meta_c1
    t2c1 = parameters.t2c1
    s = parameters.s
    nbins = len(t2c1) // 2 + 1

    t2c1.values = np.sort(t2ca.values)
    meta_model = MetaSignalDetectionModel(nbins, meta_d1, meta_c1, t2c1, s)

    output = {
        "dependent": meta_model.t2_probs(),
    }

    return output