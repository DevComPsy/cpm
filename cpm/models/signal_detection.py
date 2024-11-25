import numpy as np
from scipy.stats import norm
import pandas as pd


__all__ = ['MetaSignalDetectionHelper', 'MetaSignalDetectionModel']




class MetaSignalDetectionHelper:

    def __init__(self, data = None, nbins = 4, s = 1, fncdf = norm.cdf, fninv = norm.ppf, apply_adjustment = False):

        self.data = data
        self.nbins = nbins
        self.s = s
        self.fncdf = fncdf
        self.fninv = fninv
        self.apply_adjustment = apply_adjustment
        if data is not None:
            self.counts, self.nR_S1, self.nR_S2 = self.bin_confidence()
            self.d1 = self.compute_d1()
            self.c1 = self.compute_c1()
            self.t1c1 = self.c1[self.nbins-1]
            self.t2c1 = self.c1 - self.t1c1
            self.t2c1 = np.delete(self.t2c1, self.nbins-1)
            self.data_pandas = self.create_df()
    
    def bin_confidence(self):
        
        data = self.data
        conf = data['confidence']
        assert conf.all() >= 0 and conf.all() <= 1, "Confidence values must be between 0 and 1"
        response = data['response']
        stimulus = data['stimulus']

        bin_edges = np.linspace(0, 1, self.nbins+1)
        counts = np.zeros((2, 2, self.nbins))

        for S in range(2):
            for R in range(2):
                counts[S, R, :] = np.histogram(conf[stimulus == S][response == R], bins = bin_edges)[0]
        
        nR_S1 = np.concatenate([counts[0,0,::-1], counts[0,1,:]]) + 1/(2*self.nbins-1) if self.apply_adjustment else 0
        nR_S2 = np.concatenate([counts[1,0,::-1], counts[1,1,:]]) + 1/(2*self.nbins-1) if self.apply_adjustment else 0

        if (nR_S1 == 0).any() or (nR_S2 == 0).any():
            raise ValueError("There are bins with zero counts. Please check the data") # TODO: replace with warning

        return counts, nR_S1, nR_S2
    
    def create_df(self):

        observed = np.array([
                self.nR_S1[:self.nbins][::-1], 
                self.nR_S1[self.nbins:],
                self.nR_S2[self.nbins:],
                self.nR_S2[:self.nbins][::-1],
        ])

        print(observed)
        
        df = pd.DataFrame({
            'nbins': [self.nbins],
            'observed': [observed],
        })

        print(df)

        return df
    
    def compute_d1(self):

        ratingHR  = []
        ratingFAR = []
        for c in range(1, int(2*self.nbins)):
            ratingHR.append(sum(self.nR_S2[c:]) / sum(self.nR_S2))
            ratingFAR.append(sum(self.nR_S1[c:]) / sum(self.nR_S1))
        
        d1 = 1 / self.s * (self.fninv( ratingHR[self.nbins-1] ) - self.fninv( ratingFAR[self.nbins-1] ))
        return d1
    
    def compute_c1(self):
            
        nR_S1 = self.nR_S1
        nR_S2 = self.nR_S2
        nR_S1 = nR_S1 / nR_S1.sum()
        nR_S2 = nR_S2 / nR_S2.sum()
        c1 = -0.5 * (norm.ppf(nR_S1[1:]) + norm.ppf(nR_S2[1:]))
        return c1


class MetaSignalDetectionModel:

    def __init__(self, nbins, d1, t1c1, meta_d1, t2c1, s = 1, fncdf = norm.cdf, fninv = norm.ppf):

        assert len(t2c1) == 2*nbins - 2, "The parameters for the type 2 criteria must be twice the number of confidence bins minus 2"

        self.d1 = d1
        self.t1c1 = t1c1
        self.meta_d1 = meta_d1
        self.meta_c1 = meta_d1 * t1c1 / d1
        self.t2c1 = t2c1
        self.s = s
        self.fncdf = fncdf
        self.fninv = fninv
        self.S1mu = -self.meta_d1 / 2 - self.meta_c1
        self.S1sd = 1
        self.S2mu = self.meta_d1 / 2 - self.meta_c1
        self.S2sd = 1 / self.s
        self.nbins = nbins

    def t2_probs(self):

        t2c1x = [-np.inf]
        t2c1x.extend(self.t2c1["value"][:(self.nbins-1)])
        t2c1x.append(0)
        t2c1x.extend(self.t2c1["value"][(self.nbins-1):])
        t2c1x.append(np.inf)

        C_area_rS1 = self.fncdf(0, self.S1mu, self.S1sd)
        I_area_rS1 = self.fncdf(0, self.S2mu, self.S2sd)
        
        C_area_rS2 = 1 - self.fncdf(0, self.S2mu, self.S2sd)
        I_area_rS2 = 1 - self.fncdf(0, self.S1mu, self.S1sd)

        prC_rS1 = [
            ( self.fncdf(t2c1x[i+1], self.S1mu, self.S1sd) - self.fncdf(t2c1x[i], self.S1mu, self.S1sd) ) / C_area_rS1 
            for i in range(self.nbins)
            ]
        prI_rS1 = [
            ( self.fncdf(t2c1x[i+1], self.S2mu, self.S2sd) - self.fncdf(t2c1x[i], self.S2mu, self.S2sd) ) / I_area_rS1 
            for i in range(self.nbins)
            ]

        prC_rS2 = [
            ( (1-self.fncdf(t2c1x[self.nbins+i], self.S2mu, self.S2sd)) - (1-self.fncdf(t2c1x[self.nbins+i+1], self.S2mu, self.S2sd)) ) / C_area_rS2
            for i in range(self.nbins)
            ]
        prI_rS2 = [
            ( (1-self.fncdf(t2c1x[self.nbins+i], self.S1mu, self.S1sd)) - (1-self.fncdf(t2c1x[self.nbins+i+1], self.S1mu, self.S1sd)) ) / I_area_rS2
            for i in range(self.nbins)
            ]
        
        return np.array([prC_rS1, prI_rS1, prC_rS2, prI_rS2])
        


class SignalDetection:

    def __init__(self, stimulus, response):

        self.stimulus = stimulus
        self.response = response
        self.hit_rate, self.false_alarm_rate = self._rates()
        self.d_prime = self._d_prime(self.hit_rate, self.false_alarm_rate)
        self.c_bias = self._c_bias(self.hit_rate, self.false_alarm_rate)
    
    def _rates(self):

        hits = ( (self.stimulus == 1) & (self.response == 1) ).sum()
        misses = ( (self.stimulus == 1) & (self.response == 0) ).sum()
        false_alarms = ( (self.stimulus == 0) & (self.response == 1) ).sum()
        correct_rejections = ( (self.stimulus == 0) & (self.response == 0) ).sum()

        hit_rate = hits / (hits + misses)
        false_alarm_rate = false_alarms / (false_alarms + correct_rejections)

        return hit_rate, false_alarm_rate
    
    def _d_prime(self, hit_rate, false_alarm_rate):

        # calculate z-scores
        z_hit = norm.ppf(hit_rate)
        z_fa = norm.ppf(false_alarm_rate)

        # calculate d'
        d_prime = z_hit - z_fa

        return d_prime
    
    def _c_bias(self, hit_rate, false_alarm_rate):

        # calculate z-scores
        z_hit = norm.ppf(hit_rate)
        z_fa = norm.ppf(false_alarm_rate)

        # calculate c
        c = -0.5 * (z_hit + z_fa)

        return c