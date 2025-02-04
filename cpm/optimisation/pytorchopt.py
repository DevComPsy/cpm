import numpy as np
from scipy.stats import uniform
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch as tc
from torch.distributions import Normal

__all__ = ["PtAdam"]



class PtAdam:

    def __init__(
        self,
        parameters=None,
        data=None,
        number_of_starts=1,
        device=None,
        prior=False,
        ppt_identifier="ppt",
        constraints=None,
        constraint_penalty=1e6,
        verbose=True,
        nbins=4,
        d1s=1,
        t1c1s=0,
        lr=1e-3,
        maxiters = 10000,
    ):
        self.data = tc.tensor(data, device=device).unsqueeze(-1).expand(-1,-1,-1,number_of_starts)
        self.data = tc.stack([
            self.data[:,0,:nbins],
            self.data[:,1,:nbins],
            self.data[:,1,nbins:],
            self.data[:,0,nbins:]
        ], dim=1)

        self.maxiters = maxiters
        
        self.device = tc.device(device) if device is not None else tc.device("cuda" if tc.cuda.is_available() else "cpu")

        self.N = self.data.shape[0]
        self.prior = prior
        self.ns = number_of_starts
        self.nbins = data.shape[-1] // 2

        self.params = parameters
        # self.parameter_names = self.params.free()
        self.priors = [parameters[key].prior for key in parameters.keys() if hasattr(parameters[key], "prior")]
        self.bounds = parameters.bounds()

        self.priors =[uniform(loc=low, scale=high-low) for low, high in zip(*self.bounds)]
            
        self.nparams = len(self.priors)

        self._min_loss = tc.ones(self.N, device=self.device).double() * tc.inf
        
        self.d1s = tc.tensor(d1s, device=self.device).squeeze().unsqueeze(1).unsqueeze(2).expand(self.N, self.nbins, self.ns)
        self.t1c1s = tc.tensor(t1c1s, device=self.device).squeeze().unsqueeze(1).unsqueeze(2).expand(self.N, self.nbins, self.ns)

        self._inf = tc.tensor([[[tc.inf]]], device=self.device).expand(self.N, 1, self.ns)
        self._zero = tc.zeros((self.N, 1, self.ns), device=self.device)
        self.__zero = tc.zeros((self.N, self.nbins, self.ns), device=self.device)
        
        self.verbose = verbose

        self.lr = lr
        self.x = self._init_params()
        self.optimizer = tc.optim.Adam([self.x], lr=self.lr)
        self.scheduler = tc.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)

    def _init_params(self):
        p = np.zeros((self.N, self.nparams, self.ns))
        for i, prior in enumerate(self.priors):
            p[:,i] = prior.rvs(size=(self.N, self.ns))
        p[:,1:] = np.sort(p[:,1:], axis=1)
        init_params = tc.tensor(p, device=self.device, requires_grad=True)
        return init_params
    
    def predict(self, parameters):

        meta_d1s = parameters[:,:1]
        t2c1 = parameters[:,1:]
        const_crits = tc.clone((meta_d1s / self.d1s * self.t1c1s))
        S1mu = -meta_d1s / 2 - const_crits
        S2mu = meta_d1s / 2 - const_crits
        
        t2c1x = tc.concat([
            -self._inf,
            t2c1[:,:self.nbins-1],
            self._zero,
            t2c1[:,self.nbins-1:],
            self._inf
        ], dim=1)

        dist_S1 = Normal(S1mu, 1)
        dist_S2 = Normal(S2mu, 1)

        C_area_rS1 = dist_S1.cdf(self.__zero)
        I_area_rS1 = dist_S2.cdf(self.__zero)
        
        C_area_rS2 = 1 - dist_S2.cdf(self.__zero)
        I_area_rS2 = 1 - dist_S1.cdf(self.__zero)

        for area in [C_area_rS1, I_area_rS1, C_area_rS2, I_area_rS2]:
            area[tc.isclose(area, self.__zero.double())] = 1e-10
        
        prC_rS1 = (dist_S1.cdf(t2c1x[:,1:self.nbins+1]) - dist_S1.cdf(t2c1x[:,:self.nbins])) / C_area_rS1
        prI_rS1 = (dist_S2.cdf(t2c1x[:,1:self.nbins+1]) - dist_S2.cdf(t2c1x[:,:self.nbins])) / I_area_rS1
        prC_rS2 = ((1-dist_S2.cdf(t2c1x[:,self.nbins:-1])) - (1-dist_S2.cdf(t2c1x[:,self.nbins+1:]))) / C_area_rS2
        prI_rS2 = ((1-dist_S1.cdf(t2c1x[:,self.nbins:-1])) - (1-dist_S1.cdf(t2c1x[:,self.nbins+1:]))) / I_area_rS2

        assert all([~tc.isnan(pr).any() for pr in [prC_rS1, prI_rS1, prC_rS2, prI_rS2]]), "NaNs in probabilities."

        probs = tc.stack([prC_rS1, prI_rS1, prC_rS2, prI_rS2], dim=1)

        return probs
    
    def loss(self):

        probs = self.predict(self.x)
        probs = probs.clip(1e-10, 1)

        logs = tc.log(probs)
        assert all([~tc.isnan(log).any() for log in logs]), "NaNs in logs."

        loss = -tc.sum(self.data * logs, dim=(1,2))
        min_loss, idcs = tc.min(loss, dim=-1)
        mask = min_loss < self._min_loss
        self._min_loss[mask] = min_loss[mask]
        self.best_params = self.x[tc.arange(self.N),:,idcs].detach()
        loss = tc.mean(loss)

        # Add penalty for violating ascending order of t2c1
        t2c1 = self.x[:, 1:]  # Extract t2c1 values (excluding the first parameter, which is meta_d)
        
        # Calculate the penalty for non-ascending t2c1 values
        penalty = tc.sum(tc.relu(t2c1[:, :-1] - t2c1[:, 1:]))  # Penalty when t2c1[i] > t2c1[i+1]

        # Scale the penalty (you can adjust this based on your needs)
        penalty_weight = 1e6  # Increase if needed to enforce stricter ordering
        loss += penalty_weight * penalty

        return loss
    
    def closure(self):
        self.optimizer.zero_grad()
        self._loss = self.loss()
        self._loss.backward()
        return self._loss
    
    def step(self):
        self.closure()
        self.optimizer.step()
        self.scheduler.step()
        return self._loss.item()
    
    def run(self, patience=1000, tolerance=1e-3):
        best_loss = tc.ones(self.N, device=self.device).double() * tc.inf
        patience_counter = 0
        
        for i in range(self.maxiters):
            print(f"Iteration {i}")
            self._loss = self.step()
            if self.verbose:
                print(f"Loss: {tc.mean(self._min_loss)}")
            
            if (tc.mean(self._min_loss) < tc.mean(best_loss) - tolerance):
                best_loss = self._min_loss.clone()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping at iteration {i}")
                break
    
    def export(self):
        
        # get the values of the parameters with the minimum loss
        probs = self.predict(self.x)
        probs = probs.clip(1e-10, 1)
        min_nLL, idcs = tc.min((-self.data * tc.log(probs)).sum(dim=(1,2)), dim=-1)
        print(min_nLL)

        res = self.best_params.numpy()
        res = np.concatenate([
            self.d1s[:,0,0].numpy().reshape(-1,1),
            self.t1c1s[:,0,0].numpy().reshape(-1,1),
            res
        ], axis=1)

        cols = ["d1", "t1c1", "meta_d"] + [f"t2c1_{i}" for i in range(2*self.nbins-2)]
        
        results = pd.DataFrame(res, columns=cols)
        results['nLL'] = min_nLL.detach().numpy()
        
        return results




