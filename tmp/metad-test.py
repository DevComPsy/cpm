# from cpm.applications.signal_detection import metad_generator, d_prime_calculator
from cpm.generators import Parameters
import numpy as np
from prettyformatter import pprint as pp
import pandas as pd
from scipy.stats import norm
# Example usage
parameters = Parameters(
    bins=4,
    d_prime=2,
    s=1,
    meta_d=0.5,
    criterion_type1=0.2,
    criterion_type2=np.array([-1.5, -0.5, 0, 0.25, 0.5, 1.5]),
)
result = metad_generator(parameters)
pp(result)

flem = metad_flem(parameters.meta_d.value, parameters.criterion_type2.value, parameters.bins.value, parameters.d_prime.value, parameters.criterion_type1.value, parameters.s.value, norm.cdf)
pp(flem)



def metad_flem(metad, criterions, nRatings, d1, t1c1, s, fncdf):
    meta_d1 = metad
    t2c1    = criterions

    # define mean and SD of S1 and S2 distributions
    # adjust so that the type 1 criterion is set at 0
    # (this is just to work with optimization toolbox constraints...
    #  to simplify defining the upper and lower bounds of type 2 criteria)
    constant_criterion = meta_d1 * (t1c1 / d1)
    S1mu = -meta_d1/2
    S1mu = S1mu - constant_criterion
    S1sd = 1
    S2mu = meta_d1/2
    S2sd = S1sd/s
    S2mu = S2mu - constant_criterion

    t1c1 = 0

    # set up MLE analysis
    # get type 2 response counts
    # S1 responses
    # get type 2 probabilities
    C_area_rS1 = fncdf(t1c1,S1mu,S1sd)
    I_area_rS1 = fncdf(t1c1,S2mu,S2sd)
    
    C_area_rS2 = 1-fncdf(t1c1,S2mu,S2sd)
    I_area_rS2 = 1-fncdf(t1c1,S1mu,S1sd)
    
    t2c1x = [-np.inf]
    t2c1x.extend(t2c1[0:(nRatings-1)])
    t2c1x.append(t1c1)
    t2c1x.extend(t2c1[(nRatings-1):])
    t2c1x.append(np.inf)

    prC_rS1 = [( fncdf(t2c1x[i+1],S1mu,S1sd) - fncdf(t2c1x[i],S1mu,S1sd) ) / C_area_rS1 for i in range(nRatings)]
    prI_rS1 = [( fncdf(t2c1x[i+1],S2mu,S2sd) - fncdf(t2c1x[i],S2mu,S2sd) ) / I_area_rS1 for i in range(nRatings)]

    prC_rS2 = [( (1-fncdf(t2c1x[nRatings+i],S2mu,S2sd)) - (1-fncdf(t2c1x[nRatings+i+1],S2mu,S2sd)) ) / C_area_rS2 for i in range(nRatings)]
    prI_rS2 = [( (1-fncdf(t2c1x[nRatings+i],S1mu,S1sd)) - (1-fncdf(t2c1x[nRatings+i+1],S1mu,S1sd)) ) / I_area_rS2 for i in range(nRatings)]

    # combine all pr somethings into a dictionary
    pr_dict = {
        'pr': np.array([prC_rS1, prI_rS1, prC_rS2, prI_rS2]),
        't2c1x': np.asarray(t2c1x),
        't2c1': t2c1,
        'S1mu': S1mu,
        'S1sd': S1sd,
        'S2mu': S2mu,
        'S2sd': S2sd,
        'area_under_curve' : np.array([C_area_rS1, I_area_rS1, C_area_rS2, I_area_rS2]),
        'constant_criterion': constant_criterion,
    }

    return pr_dict



import math
# Example data for nR_S1 and nR_S2
nR_S1 = [50, 30, 20, 10, 5, 2, 1, 0]
nR_S2 = [0, 1, 2, 5, 10, 20, 30, 50]

## reversed cumulative sums
nR_S1 = nR_S1[::-1]
nR_S2 = nR_S2[::-1]

## TODO: check against original code
## TODO:: implement in the metad_generator
## FIXME: scaled cumulative sums have to be reversed (ordered)
## FIXME: has extra value at beginning/end
# Convert to numpy arrays
nR_S1 = np.array(nR_S1)
nR_S2 = np.array(nR_S2)
hr = np.cumsum(nR_S1) / np.sum(nR_S1)
fa = np.cumsum(nR_S2) / np.sum(nR_S2) ## has extra

hr = hr[::-1]
hr = hr[1:]
fa = fa[::-1]
fa = fa[1:]


# Number of ratings
nRatings = len(nR_S1) // 2

# Calculate ratingHR and ratingFAR
ratingHR = []
ratingFAR = []
for c in range(1, int(nRatings * 2)):
    ratingHR.append(sum(nR_S2[c:]) / sum(nR_S2))
    ratingFAR.append(sum(nR_S1[c:]) / sum(nR_S1))

# Print the results
print(np.asarray(ratingHR))
print(np.asarray(ratingFAR))
s=1
c1 = (-1/(1+s)) * ( norm.ppf( ratingHR ) + norm.ppf( ratingFAR ) )
c1[3]
# set up initial guess at parameter values
ratingHR  = []
ratingFAR = []
for c in range(1,int(nRatings*2)):
    ratingHR.append(sum(nR_S2[c:]) / sum(nR_S2))
    ratingFAR.append(sum(nR_S1[c:]) / sum(nR_S1))