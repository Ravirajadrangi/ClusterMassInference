#!/u/ki/swmclau2/PYENV/bin/python
#@Author Sean McLaughlin
desc ='''
This module will eventually become part of the larger Pangloss structure.

I'm naively porting my work from jupyter notebooks to a .py file, for easier running on large clusters. I hope to work on better code
structure in the future.

This module attempts to infer the richness mass relation for galaxy clusters. I was agiven a new model by my advisor, as there were some complications
with the other one I was working with. I've successfully recovered model parameters in the case where the masses are a measured quantity. I wish now to
use MCMC to integrate over uncertainty in mass. This will be a large job I'll have to run on a SLAC cluster.
'''

from argparse import ArgumentParser
parser = ArgumentParser(description = desc)

parser.add_argument('nWalkers', metavar = 'nWalkers', type = int, help =\
                    'Number of walkers to use in the MCMC sampler. Appropriate values are around 1000.')

parser.add_argument('nSteps', metavar = 'nSteps', type = int, help =\
                    'Number of steps for the sampler to take. Appropriate values are 100-500')

parser.add_argument('nCores', metavar = 'nCores', type = int, help=\
                    'Number of cores to allow the sampler to use. Limited by machine maximum.')

#parser.add_argument('--noDisplay', dest = 'noDisplay', action = 'store_true', help =\
#                    'If used, will assume the job is running on a machine with no display and will make adjustments.')

args = parser.parse_args()

nWalkers = args.nWalkers
nSteps = args.nSteps
nCores = args.nCores

if any(x < 1 for x in [nWalkers, nSteps, nCores]):
    print 'Invalid input.'
    from sys import exit
    exit(-1)

from multiprocessing import cpu_count
maxCores = cpu_count()
if nCores>maxCores:
    print 'WARNING: Specified number of cores greater than total available.'
    nCores = maxCores

from astropy.io import fits
import numpy as np
from scipy.stats import linregress, gamma, expon, norm
import emcee as mc

#dataDir = '/home/sean/Data/BuzzardSims/'
dataDir = '/nfs/slac/g/ki/ki19/des/erykoff/clusters/mocks/Buzzard/buzzard-1.1/des_y5/redmapper_v6.4.7/halos/'
hdulist = fits.open(dataDir+'buzzard-v1.1-y5_run_00340_lambda_chisq.fit')
data = hdulist[1].data

mass = data['M200']
redshifts = data['Z']

mass = mass[redshifts<0.9]#Take only the first bin
redshifts = redshifts[redshifts<0.9]

#true params
M_piv = 2.35e14
logL0_true = 3.141
a_true, b_true = 0.842, -0.03
B_l_true = 0.642
sigma_l_true = 0.184

_A_lam = lambda a, b, z : a*pow((1+z)/1.3, b)

#forward model
def logLam(logL0, a, b, B_l, M, z):
    A_lam = _A_lam(a,b,z)
    return logL0+A_lam*np.log(M/M_piv)+B_l*np.log((1+z)/1.3)

def logLamSample(logL0, a, b, B_l, sigma_l, M, z):
    return norm.rvs(loc = logLam(logL0, a, b, B_l, M, z), scale = sigma_l)

#sample "measured" richnesses
logRichness = logLamSample(logL0_true, a_true, b_true, B_l_true, sigma_l_true, redshifts, mass)

vals = {}
z = [0.23, 1.5]
vals['Mp'] = [2.0e14, 1.0e14]#*10^14
vals['A'] = [1.944, 0.293]
vals['B1'] = [1.97, 3.07]
vals['B2'] = [0.7, 1.2]
vals['B3'] = [0.40, 0.73]

#tools for mass function
#interpolates naively between the points given in the original paper
z_params = {}
for key, val in vals.iteritems():
    slope, intercept, r, p, stderr = linregress(z, val)
    z_params[key] = (slope, intercept)

#return the values of the parameters at a given z.
getMassParams = lambda z : {key:val[0]*z+val[1] for key,val in z_params.iteritems()}

def log_n_approx(m,z):
    params = getMassParams(z)
    return -1*(params['B1']*(m/params['Mp']) \
                    +0.5*params['B2']*(m/params['Mp'])**2 \
                    +0.166*params['B3']*(m/params['Mp'])**3)+np.log(params['A'])

def log_prior(logL0, a,b,B_l, sigma, M):

    if any(x<0 for x in (logL0,sigma)):
        return -np.inf

    if np.any(M<0): #masses have to be positive
        return -np.inf

    t1 = np.arctan(B_l)
    t2 = np.arctan(a)
    #if t<0 or t>np.pi/2:
    if any(x< -np.pi/2 or x> np.pi/2 for x in (t1,t2)):
        return -np.inf

    #Hyperparameters
    lambda_logL0 = 1.0
    sigma_a, sigma_b = 1,1

    p = 0
    #Exponential in logL0
    p+= expon.logpdf(logL0, scale = 1/lambda_logL0)
    #Uniform in arctan(B_l) and arctan(a)
    p+=2*np.log(2/np.pi)
    #flat prior for b
    #Have not idea what it would be, start with nothing
    #p+=0

    #Inv Gamma for sigma
    p-= gamma.logpdf(sigma**2,sigma_a, scale = sigma_b)
    return p

#TODO log_n_approx and logLam both divide mass by a different pivot. Should I use the same?
#I should also consider pre-dividing by the pivot to avoid repeated calculations.
def log_liklihood(logL0, a,b, B_l, sigma,M, z, logRich):
    p = 0
    #mass function
    p+= np.sum(log_n_approx(M,z))#not normalized, if that's a problem i can approximate it.
    #lillihood of richness
    p+=np.sum(norm.logpdf(logRich, loc =logLam(logL0, a, b, B_l, M, z), scale = sigma))
    return p

def log_posterior(theta,z,logRich):
    #print theta
    logL0, a,b, B_l, sigma = theta[:5]
    M = theta[5:]
    p = log_prior(logL0, a,b, B_l, sigma, M)
    if np.isfinite(p):
        p+=log_liklihood(logL0, a,b,B_l, sigma,M, z, logRich)
    return p

#set up sampler
ndim = 5 + logRichness.shape[0]
if 2*ndim>nWalkers:
    nWalkers = 2*ndim
    print 'nWalkers changed to %d to fit number of dimensions'%nWalkers

logL0_a, logL0_b = 1, 1 # Gamma
a_mean, a_spread = 1, 1.5
b_mean, b_spread = 0, .5
B_l_mean, B_l_spread = 1, 1.5
sigma_a, sigma_b = 1, 1 #Gamma
mass_mean, mass_spread = 14, .5 #logNormal

pos0 = np.zeros((nWalkers, ndim))
for row in pos0:

    row[0] = gamma.rvs(logL0_a, scale = logL0_b)
    row[1] = norm.rvs(loc = a_mean, scale = a_spread)
    row[2] = norm.rvs(loc = b_mean, scale = b_spread)
    row[3] = norm.rvs(loc = B_l_mean, scale = B_l_mean)
    row[4] = gamma.rvs(sigma_a, scale = sigma_b)
    row[5:] = 10**(norm.rvs(loc = mass_mean, scale = mass_spread, size = logRichness.shape[0]))

sampler = mc.EnsembleSampler(nWalkers, ndim, log_posterior, args=[redshifts, logRichness],threads = nCores)
nburn = int(nSteps)/10

sampler.run_mcmc(pos0, nSteps)

chain = sampler.chain[:,nburn:, :].reshape((-1, ndim))
sampler.pool.terminate()#there's a bug in emcee that creates daemon threads. This kills them.
del(sampler)

MAP = chain[:, :5].mean(axis = 0)
print MAP
labels = ['logL0', 'a', 'b','B_l','sigma']
print '\tMCMC\tTrue'
for label, val, truth in zip(labels, MAP, [logL0_true, a_true, b_true, B_l_true, sigma_l_true]):
    print '%s:\t%.3f\t%.3f'%(label, val, truth)

np.savetxt('chain_%dw_%ds.gz'%(nWalkers, nSteps), chain)
