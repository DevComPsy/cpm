import numpy as np

__all__ = ["RapidAttentionShift"]


class RapidAttentionShift:
    """
        RapidAttentionShift implements the rapid attention shift from models such as EXIT (Kruschke, 2001) and its simplified version (Paskewitz & Jones, 2020). The rapid attention shift mechanism allows the model to quickly adjust attention to stimuli based on prediction errors, enabling faster learning, and is an essential component in models of associative learning that incorporate attention mechanisms in the tradition of Mackintosh (1975).
        
        Parameters
        ----------
        weights : array-like
            Weight matrix of shape (n_outcomes, n_stimuli).
        predictions : array-like
            Model predictions, a 1D array of length n_outcomes.
        input : array-like
            Stimulus input, a 1D array of length n_stimuli.
        error : array-like
            Prediction errors, a 1D array of length n_outcomes.
        gain : array-like
            Current attention gain values, a 1D array of length n_stimuli.
        pnorm : float
            Normalisation constant.
        P : float
            Power parameter.
        rho : float
            Attention shift rate (step size).
        **kwargs : dict, optional
            Additional keyword arguments.

        See Also
        --------
        [cpm.models.activation.CompetitiveGating][cpm.models.activation.CompetitiveGating] : A class representing competitive attention gating, which can be used in conjunction with RapidAttentionShift to model attention mechanisms.

        Notes
        -----
        The rapid attention shift is defined as the change in attention gain for stimulus $j$ on a given iteration, computed as:

        $$
            \\Delta g_j' = \\rho ||g'||_p^{-1} \\circ \\sum_{k} \\delta_k (W_{ki}s_{i} - {a'_{i}}^{p-1}{z'_{k}}^{T})
        $$

        where $a_i$ is the normalised attention gain for stimulus $i$ and $g$ is the attention gain vector calculated from saliences, $\\eta$, by $g_i = s_i*\\eta_i$. The $\\circ$ denotes element-wise multiplication. The normalised attention gain is computed as:

        $$
        a_i = \\frac{g_i}{||g||_p} \\
        $$

        The model prediction $z_k$ is computed as:
        
        $$
        z_k = \\sum_{i} W_{ki} s_i a_i
        $$
        
        Error is computed as:
        
        $$
        \\delta_k = t_k - z_k
        $$
        
        where $t_k$ is the target outcome for outcome $k$ and $z_k$ is the model prediction for outcome $k$. The attention shift is a non-linear function of attention gain. This means that the attention shift has to be computed iteratively, with the attention gain, prediction error, and model predictions being updated after each iteration until convergence. The attention shift rate, $\\rho$, controls the speed of convergence. Previous research (Kruschke, 2001) set the number of iterations to 10, but this can be adjusted based on the specific use case and convergence criteria. Here we only implement a single iteration of the attention shift, and the user can call the function multiple times to compute the attention shift iteratively until convergence.
        
        References
        ----------
        
        Kruschke, J. K. (2001). Toward a Unified Model of Attention in Associative Learning. Journal of Mathematical Psychology, 45(6), 812–863. https://doi.org/10.1006/jmps.2000.1354
        
        Mackintosh, N. J. (1975). A theory of attention: variations in the associability of stimuli with reinforcement. Psychological review, 82(4), 276.

        Paskewitz, S., & Jones, M. (2020). Dissecting EXIT. Journal of Mathematical Psychology, 97, 102371. https://doi.org/10.1016/j.jmp.2020.102371

    """

    def __init__(
        self,
        weights=None,
        predictions=None,
        input=None,
        error=None,
        gain=None,
        pnorm=1.0,
        P=2.0,
        rho=0.1,
        **kwargs,
    ):
        self.weights = np.asarray(weights, dtype=float)
        self.predictions = np.asarray(predictions, dtype=float)
        self.input = np.asarray(input, dtype=float)
        self.error = np.asarray(error, dtype=float)
        self.gain = np.asarray(gain, dtype=float)
        self.pnorm = float(pnorm)
        self.P = float(P)
        self.rho = float(rho)
        self.delta_gain = None

    def compute(self):
        """
        Compute the attention gain update.

        Returns
        -------
        ndarray
            Gain update, a 1D array of length n_stimuli.
        """
        activations = self.weights * self.input
        a_power = self.gain ** (self.P - 1)
        attention = np.outer(self.predictions, a_power)
        out = (activations - attention) * self.error[:, np.newaxis]
        out_sum = out.sum(axis=0)
        self.delta_gain = self.rho * (self.pnorm**-1) * out_sum
        return self.delta_gain

    def reset(self):
        self.delta_gain = None

    def __repr__(self):
        return (
            f"RapidAttentionShift(gain={self.gain},\n"
            f" pnorm={self.pnorm}, P={self.P}, rho={self.rho})"
        )

    def __str__(self):
        return self.__repr__()

    def __call__(self):
        return self.compute()
