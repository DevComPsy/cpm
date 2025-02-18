from audioop import mul
import numpy as np
from scipy.stats import multivariate_normal
from ..core.optimisers import LinearConstraint
from ..generators import Parameters, Value
from ..models import MetaSignalDetectionHelper, MetaSignalDetectionModel


def metacognitionSDT_initparams(
    data=None,
    binned_data=None,
    nbins=4,
    s=1,
    apply_adjustment=False,
    adjustment_val=None,
    ppt_identifier="ppt",
):

    assert (
        data is not None or binned_data is not None
    ), "Either data or binned_data must be provided."

    helper = MetaSignalDetectionHelper(
        data=data,
        nbins=nbins,
        s=s,
        apply_adjustment=apply_adjustment,
        adjustment_val=adjustment_val,
        ppt_identifier=ppt_identifier,
    )
    if data is None:
        helper.nR_S1 = np.array(binned_data["nR_S1"])
        helper.nR_S2 = np.array(binned_data["nR_S2"])
        helper.ratingHR, helper.ratingFAR = helper.rates()
        helper.d1 = helper.compute_d1()
        helper.c1 = helper.compute_c1()
        helper.t1c1 = helper.c1[helper.nbins - 1]
        helper.t2c1 = helper.c1 - helper.t1c1
        helper.t2c1 = np.delete(helper.t2c1, helper.nbins - 1)
        helper.data_pandas = helper.create_df()

    params = {
        int(ppt): Parameters(
            meta_d1=Value(
                value=helper.d1[idx],
                lower=-1.5,
                upper=3.5,
                prior="norm",
                args={"mean": 1, "std": 2},
            ),
            t2c1=Value(
                value=helper.t2c1[idx],
                lower=-5,
                upper=5,
                prior=multivariate_normal,
                args={
                    "mean": np.delete(np.linspace(-4, 4, 2 * nbins - 1), nbins - 1),
                    "cov": 3 * np.eye(2 * nbins - 2),
                },
            ),
            t1c1=helper.t1c1[idx],
            s=s,
            d1=helper.d1[idx],
            nbins=nbins,
        )
        for idx, ppt in enumerate(helper.ppts)
    }

    # Generate matrices A and b based on nbins
    A = np.zeros((2 * nbins - 1, 2 * nbins - 1))
    for i in range(2 * nbins - 3):
        A[i, i + 1 : i + 3] = [-1, 1]
    A[-2, nbins - 1] = -1
    A[-1, nbins] = 1
    b = np.zeros(2 * nbins - 1)

    constraints = LinearConstraint(A=A, b=b)

    return params, helper.d1, helper.t1c1, helper.data_pandas, constraints, helper
