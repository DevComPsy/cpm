import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from audioop import mul
from scipy.stats import multivariate_normal, norm
from .core.optimisers import LinearConstraint
from .generators import Parameters, Value
from .models import MetaSignalDetectionHelper, MetaSignalDetectionModel

__all__ = [
    "MetaSignalDetectionHelper",
    "MetaDetectionModel",
    "metacognitionSDT_initparams",
]


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
        helper.d_prime = helper.compute_d_prime()
        helper.c1 = helper.compute_c1()
        helper.criterion_type1 = helper.c1[helper.nbins - 1]
        helper.criterion_type2 = helper.c1 - helper.criterion_type1
        helper.criterion_type2 = np.delete(helper.criterion_type2, helper.nbins - 1)
        helper.data_pandas = helper.create_df()

    params = {
        int(ppt): Parameters(
            meta_d=Value(
                value=helper.d_prime[idx],
                lower=-1.5,
                upper=3.5,
                prior="norm",
                args={"mean": 1, "std": 2},
            ),
            criterion_type2=Value(
                value=helper.criterion_type2[idx],
                lower=-5,
                upper=5,
                prior=multivariate_normal,
                args={
                    "mean": np.delete(np.linspace(-4, 4, 2 * nbins - 1), nbins - 1),
                    "cov": 3 * np.eye(2 * nbins - 2),
                },
            ),
            criterion_type1=helper.criterion_type1[idx],
            s=s,
            d_prime=helper.d_prime[idx],
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

    return (
        params,
        helper.d_prime,
        helper.criterion_type1,
        helper.data_pandas,
        constraints,
        helper,
    )


class MetaSignalDetectionHelper:
    """
    Helper class for models calculating meta-d based on Signal Detection Theory. It computes the necessary parameters for the MetaSignalDetectionModel class.
    """

    def __init__(
        self,
        data=None,
        nbins=4,
        s=1,
        fncdf=norm.cdf,
        fninv=norm.ppf,
        apply_adjustment=False,
        adjustment_val=None,
        ppt_identifier="ppt",
    ):

        self.data = data
        self.nbins = nbins
        self.s = s
        self.fncdf = fncdf
        self.fninv = fninv
        self.apply_adjustment = apply_adjustment
        self.adjustment_val = (
            adjustment_val if adjustment_val is not None else 1 / (2 * self.nbins)
        )
        self.ppt_identifier = ppt_identifier
        self.ppts = (
            self.data[self.ppt_identifier].unique()
            if self.ppt_identifier is not None
            else [0]
        )
        if data is not None:
            self.nR_S1, self.nR_S2 = self.bin_confidence()
            self.ratingHR, self.ratingFAR = self.rates()
            self.d_prime = self.compute_d_prime()
            self.c1 = self.compute_c1()
            self.criterion_type1 = self.c1[:, self.nbins - 1, np.newaxis]
            self.criterion_type2 = self.c1 - self.criterion_type1
            self.criterion_type2 = np.delete(
                self.criterion_type2, self.nbins - 1, axis=1
            )
            self.data_pandas = self.create_df()

    def new_bin_ratings(self, slidingCon):
        """
        Bins the slidingCon array into self.nbins based on quantiles, ensuring proper assignment of binned confidence levels.

        Parameters:
            slidingCon (numpy.ndarray): Array of sliding confidence values.

        Returns:
            responseConf (numpy.ndarray): Array with binned confidence ratings.
            out (dict): Contains information about the confidence bins and rebinning status.
        """
        out = {}
        responseConf = np.zeros_like(slidingCon, dtype=int)

        # Calculate quantiles for binning
        binEdges = np.quantile(slidingCon, np.linspace(0, 1, self.nbins + 1))

        # Check for edge cases in binning
        if (
            binEdges[0] == binEdges[1]
            and binEdges[self.nbins - 1] == binEdges[self.nbins]
        ):
            raise ValueError("Bad bins!")
        elif binEdges[self.nbins - 1] == binEdges[self.nbins]:
            # Handle high-confidence issues
            hiConf = binEdges[self.nbins]
            binEdges = np.quantile(
                slidingCon[~(slidingCon == hiConf)],
                np.linspace(0, 1, self.nbins),
                method="interpolated_inverted_cdf",
            )

            temp = []
            for b in range(len(binEdges) - 1):
                temp.append(
                    (slidingCon >= binEdges[b]) & (slidingCon < binEdges[b + 1])
                )
            temp.append(slidingCon == hiConf)

            out["confBins"] = np.append(binEdges, hiConf)
            out["rebin"] = 1

        elif binEdges[0] == binEdges[1]:
            # Handle low-confidence issues
            lowConf = binEdges[0]
            temp = [slidingCon == lowConf]
            binEdges = np.quantile(slidingCon[~temp[0]], np.linspace(0, 1, self.nbins))

            for b in range(1, len(binEdges)):
                temp.append(
                    (slidingCon >= binEdges[b - 1]) & (slidingCon < binEdges[b])
                )

            out["confBins"] = np.insert(binEdges, 0, lowConf)
            out["rebin"] = 1

        else:
            temp = []
            for b in range(len(binEdges) - 2):
                temp.append(
                    (slidingCon >= binEdges[b]) & (slidingCon < binEdges[b + 1])
                )
            temp.append((slidingCon >= binEdges[-2]) & (slidingCon <= binEdges[-1]))

            out["confBins"] = binEdges
            out["rebin"] = 0

        # Assign responseConf based on bins and count each bin's size
        for b in range(self.nbins):
            responseConf[temp[b]] = b
            out.setdefault("binCount", []).append(np.sum(temp[b]))

        return responseConf, out

    def bin_confidence(self):

        nR_S1 = np.zeros((len(self.ppts), 2 * self.nbins))
        nR_S2 = np.zeros((len(self.ppts), 2 * self.nbins))

        for idx, ppt in enumerate(self.ppts):
            ppt_data = self.data[self.data[self.ppt_identifier] == ppt]
            conf = ppt_data["confidence"]
            assert (conf >= 0).all() and (
                conf <= 1
            ).all(), "Confidence values must be between 0 and 1"
            response = ppt_data["response"]
            stimulus = ppt_data["stimulus"]

            counts = np.zeros((2, 2, self.nbins))

            """
            # bin_edges = np.linspace(conf.min(), conf.max(), self.nbins+1)
            bin_edges = np.linspace(0, 1, self.nbins+1)

            # for S in range(2):
            #     for R in range(2):
            #         counts[S, R, :] = np.histogram(conf[stimulus == S][response == R], bins = bin_edges)[0]

            # nR_S1[idx] = np.concatenate([counts[0,0,::-1], counts[0,1,:]]) 
            # nR_S2[idx] = np.concatenate([counts[1,0,::-1], counts[1,1,:]])

            counts = np.zeros((2, 2, self.nbins))
            kde_points = np.linspace(0, 1, 1000)
            # kde_points = np.linspace(conf.min(), conf.max(), 1000)
            for S in range(2):
                for R in range(2):
                    # Compute kernel density estimate
                    try:
                        kde = gaussian_kde(conf[stimulus == S][response == R])
                    except:
                        raise ValueError(f"There are not diverse enough confidence values to compute the KDE. Please check the data of participant {ppt}.")
                    # kde_values = kde((bin_edges[1:] + bin_edges[:-1]) / 2)
                    kde_values = kde(kde_points)
                    # Normalize the KDE values to sum to the same total count as the original histogram
                    kde_values /= kde_values.sum()
                    binned_kde_values = np.histogram(kde_points, bins=bin_edges, weights=kde_values)[0]
                    counts[S, R, :] = binned_kde_values * ((stimulus == S) & (response == R)).sum()
                    # counts[S, R, :] = kde_values * ((stimulus == S) & (response == R)).sum()

            nR_S1[idx] = np.concatenate([counts[0,0,::-1], counts[0,1,:]]) 
            nR_S2[idx] = np.concatenate([counts[1,0,::-1], counts[1,1,:]])
        
        # print(nR_S1, nR_S2)
        """

            # Bin confidence values
            responseConf, out = self.new_bin_ratings(conf)
            for S, R in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                i_bin, bincounts = np.unique(
                    responseConf[(stimulus == S) & (response == R)], return_counts=True
                )
                i_bin = i_bin + self.nbins if R == 1 else self.nbins - i_bin - 1
                if S == 0:
                    nR_S1[idx, i_bin] = bincounts
                else:
                    nR_S2[idx, i_bin] = bincounts

        nR_S1 = nR_S1 + self.adjustment_val if self.apply_adjustment else nR_S1
        nR_S2 = nR_S2 + self.adjustment_val if self.apply_adjustment else nR_S2

        # nR_S1 += self.adjustment_val if self.apply_adjustment else 0
        # nR_S2 += self.adjustment_val if self.apply_adjustment else 0

        if (nR_S1 == 0).any() or (nR_S2 == 0).any():
            raise ValueError("There are bins with zero counts. Please check the data")

        return nR_S1, nR_S2

    def create_df(self):

        observed = [
            np.array(
                [
                    self.nR_S1[i, : self.nbins],
                    self.nR_S2[i, : self.nbins],
                    self.nR_S2[i, self.nbins :],
                    self.nR_S1[i, self.nbins :],
                ]
            )
            for i in range(len(self.ppts))
        ]

        df = pd.DataFrame(
            {
                "ppt": self.ppts,
                "nbins": [self.nbins] * len(self.ppts),
                "observed": observed,
            }
        )

        return df

    def rates(self):

        ratingHR = np.zeros((len(self.ppts), 2 * self.nbins - 1))
        ratingFAR = np.zeros((len(self.ppts), 2 * self.nbins - 1))
        for idx, ppt in enumerate(self.ppts):
            for c in range(1, int(2 * self.nbins)):
                ratingHR[idx, c - 1] = sum(self.nR_S2[idx, c:]) / sum(self.nR_S2[idx])
                ratingFAR[idx, c - 1] = sum(self.nR_S1[idx, c:]) / sum(self.nR_S1[idx])

        return np.array(ratingHR), np.array(ratingFAR)

    def plot_ROC(self):
        for idx, ppt in enumerate(self.ppts):
            fig, axs = plt.subplots()
            axs.scatter(
                self.ratingFAR[idx, : self.nbins],
                self.ratingHR[idx, : self.nbins],
                label="S2",
            )
            axs.scatter(
                self.ratingFAR[idx, self.nbins - 1 :],
                self.ratingHR[idx, self.nbins - 1 :],
                label="S1",
            )
            axs.plot([0, 1], [0, 1], "k--")
            axs.set_xlabel("False Alarm Rate")
            axs.set_ylabel("Hit Rate")
            axs.set_title(f"ROC Curve for PPT {ppt}")
            axs.legend()
            plt.show()

    def compute_d_prime(self):

        d_prime = np.zeros(len(self.ppts))
        for idx, ppt in enumerate(self.ppts):
            d_prime[idx] = 1 / self.s * self.fninv(
                self.ratingHR[idx, self.nbins - 1]
            ) - self.fninv(self.ratingFAR[idx, self.nbins - 1])
        return d_prime

    def compute_c1(self):

        c1 = np.zeros((len(self.ppts), 2 * self.nbins - 1))
        for idx, ppt in enumerate(self.ppts):
            c1[idx] = (
                -1
                / (1 + self.s)
                * (self.fninv(self.ratingHR[idx]) + self.fninv(self.ratingFAR[idx]))
            )
        return c1


class MetaSignalDetectionModel:
    """
    Class for models calculating meta-d based on Signal Detection Theory.

    Parameters
    ----------
    nbins : int
        Number of confidence bins.
    d_prime : float
        d-prime value, also called sensitivity or discriminability index. It is the distance between the two distributions (one fore each stimulus type).
    s : float
        Signal detection criterion.
    meta_d: float
        meta-d value, also called metacognitive sensitivity.
    criterion_type1 : float
        Response criteria for stimulus type 1.
    criterion_type2: float
        Response criteria for stimulus type 2.
    fncdf : function
        Cumulative distribution function. Default is scipy.stats.norm.cdf.
    fninv : function
        Inverse cumulative distribution function. Default is scipy.stats.norm.ppf.

    Returns
    -------
    cpm.applications.signal_detection.MetaSignalDetectionModel
    """

    def __init__(
        self,
        nbins,
        d_prime=None,
        s=None,
        meta_d=None,
        criterion_type1=None,
        criterion_type2=None,
        fncdf=norm.cdf,
        fninv=norm.ppf,
    ):

        if s is None:
            s = 1
        if len(criterion_type2) != 2 * nbins - 2:
            raise ValueError(
                "The parameters for the type 2 criteria must be twice the number of confidence bins minus 2"
            )

        self.d_prime = d_prime
        self.criterion_type1 = criterion_type1
        self.criterion_type2 = criterion_type2
        self.meta_d = meta_d
        self.meta_criterion = meta_d * criterion_type1 / d_prime
        self.s = s
        self.fncdf = fncdf
        self.fninv = fninvs
        self.S1mu = -self.meta_d / 2 - self.meta_criterion
        self.S1sd = 1
        self.S2mu = self.meta_d / 2 - self.meta_criterion
        self.S2sd = 1 / self.s
        self.nbins = nbins

    def compute(self):
        """
        Calculate the probability of each confidence bin given the stimulus type and response.
        """

        criterion_type2_array = (
            self.criterion_type2.value
            if hasattr(self.criterion_type2, "value")
            else self.criterion_type2
        )

        criterion_type2x = [-np.inf]
        criterion_type2x.extend(criterion_type2_array[: (self.nbins - 1)])
        criterion_type2x.append(0)
        criterion_type2x.extend(criterion_type2_array[(self.nbins - 1) :])
        criterion_type2x.append(np.inf)

        C_area_rS1 = self.fncdf(0, self.S1mu, self.S1sd)
        I_area_rS1 = self.fncdf(0, self.S2mu, self.S2sd)

        C_area_rS2 = 1 - self.fncdf(0, self.S2mu, self.S2sd)
        I_area_rS2 = 1 - self.fncdf(0, self.S1mu, self.S1sd)

        for var in [C_area_rS1, I_area_rS1, C_area_rS2, I_area_rS2]:
            if var.any() == 0:
                var += 1e-10

        prC_rS1 = [
            (
                self.fncdf(criterion_type2x[i + 1], self.S1mu, self.S1sd)
                - self.fncdf(criterion_type2x[i], self.S1mu, self.S1sd)
            )
            / C_area_rS1
            for i in range(self.nbins)
        ]
        prI_rS1 = [
            (
                self.fncdf(criterion_type2x[i + 1], self.S2mu, self.S2sd)
                - self.fncdf(criterion_type2x[i], self.S2mu, self.S2sd)
            )
            / I_area_rS1
            for i in range(self.nbins)
        ]

        prC_rS2 = [
            (
                (1 - self.fncdf(criterion_type2x[self.nbins + i], self.S2mu, self.S2sd))
                - (
                    1
                    - self.fncdf(
                        criterion_type2x[self.nbins + i + 1], self.S2mu, self.S2sd
                    )
                )
            )
            / C_area_rS2
            for i in range(self.nbins)
        ]
        prI_rS2 = [
            (
                (1 - self.fncdf(criterion_type2x[self.nbins + i], self.S1mu, self.S1sd))
                - (
                    1
                    - self.fncdf(
                        criterion_type2x[self.nbins + i + 1], self.S1mu, self.S1sd
                    )
                )
            )
            / I_area_rS2
            for i in range(self.nbins)
        ]

        probs = np.array([prC_rS1, prI_rS1, prC_rS2, prI_rS2])

        return probs
