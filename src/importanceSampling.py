#!/u/ki/swmclau2/PYENV/bin/python
#@Author Sean McLaughlin
desc ='''
This module will eventually become part of the larger Pangloss structure.

I'm naively porting my work from jupyter notebooks to a .py file, for easier running on large clusters. I hope to work on better code
structure in the future.

I'm implementing importance sampling to try to tackle this problem. It has become clear that MCMC won't be enough.
It was clear however that this is still intractable in a notebook, so I'm trying to push this to SLAC.
'''

from argparse import ArgumentParser
parser = ArgumentParser(description = desc)

parser.add_argument('nWalkers', metavar = 'nWalkers', type = int, help =\
                    'Number of walkers to use in the MCMC sampler. Appropriate values are around 1000.')

parser.add_argument('nSteps', metavar = 'nSteps', type = int, help =\
                    'Number of steps for the sampler to take. Appropriate values are 100-500')

parser.add_argument('nCores', metavar = 'nCores', type = int, help=\
                    'Number of cores to allow the sampler to use. Limited by machine maximum.')

parser.add_argument('nData', metavar = 'nData', type = int, help=\
                    'Number of mock data points to draw. ')

parser.add_argument('nSamples', metavar = 'nSamples', type = int, help=\
                    'Number of samples to draw during importance sampling.')

#parser.add_argument('--noDisplay', dest = 'noDisplay', action = 'store_true', help =\
#                    'If used, will assume the job is running on a machine with no display and will make adjustments.')

args = parser.parse_args()

nWalkers = args.nWalkers
nSteps = args.nSteps
nCores = args.nCores
nData = args.nData
nSamples = args.nSamples

if any(x < 1 for x in [nWalkers, nSteps, nCores, nData, nSamples]):
    print 'Invalid input.'
    from sys import exit
    exit(-1)

from multiprocessing import cpu_count
maxCores = cpu_count()
if nCores>maxCores:
    print 'WARNING: Specified number of cores greater than total available.'
    print 'nCores = %d'%maxCores
    nCores = maxCores

import numpy as np
from scipy.stats import linregress, gamma, expon, norm, t
import emcee as mc
from scipy.misc import logsumexp
from itertools import izip

np.random.seed(0)

#First, will need the parameters
redshift = .9 #one redshift, for now.
vals = {}
z = [0.23, 1.5]
vals['Mp'] = [2.0e14, 1.0e14]
vals['A'] = [1.944, 0.293]
vals['B1'] = [1.97, 3.07]
vals['B2'] = [0.7, 1.2]
vals['B3'] = [0.40, 0.73]

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

nDim_mass = 1
nWalkers_mass = 1000
nSteps_mass = 500
nBurn_mass = 400

Mmin, Mmax = 1e13, 5e15
pos0 = np.random.uniform(np.log10(Mmin), np.log10(Mmax), size = nWalkers_mass).reshape((nWalkers_mass, nDim_mass))

def log_p(logM, z):
    logM = logM[0]
    m = 10**logM
    if m>Mmax or m<Mmin:
        return -np.inf

    return log_n_approx(m, z)

sampler = mc.EnsembleSampler(nWalkers_mass, nDim_mass, log_p, args=[redshift],threads = nCores)

sampler.run_mcmc(pos0, nSteps_mass)

if nCores> 1:
    sampler.pool.terminate()#there's a bug in emcee that creates daemon threads. This kills them.

logMSamples = sampler.chain[:,nBurn_mass:, :].reshape((-1, nSamples))[0,:]
mSamples = 10**logMSamples

del(sampler)

#true params
M_piv = 2.35e14
logL0_true = 3.141
a_true, b_true = 0.842, -0.03
B_l_true = 0.642
sigma_l_true = 0.184

_A_lam = lambda a, b, z : a*pow((1+z)/1.3, b)

#forward model
def logLam(logL0, a, b, B_l,z, M):
    A_lam = _A_lam(a,b,z)
    return logL0+A_lam*np.log(M/M_piv)+B_l*np.log((1+z)/1.3)

def logLamSample(logL0, a, b, B_l, sigma_l,z, M):
    return norm.rvs(loc = logLam(logL0, a, b, B_l, z, M), scale = sigma_l, size = M.shape[0])
#sample "measured" richnesses
logRichness = logLamSample(logL0_true, a_true, b_true, B_l_true, sigma_l_true, redshift, mSamples)

#TODO I've relaxed a few priors, I could stand to strengthen them back up again.
def log_prior(logL0, a,b,B_l, sigma):

    if any(x<0 for x in (logL0,sigma)):
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
    p+=2*np.log(1/np.pi)
    #flat prior for b
    #Have not idea what it would be, start with nothing
    #p+=0

    #Inv Gamma for sigma
    p-= gamma.logpdf(sigma**2,sigma_a, scale = sigma_b)
    return p

#returns a logmass given a richness
def invLogLam(logL0, a, b, B_l, z, logRich):
    A_l = _A_lam(a,b,z)
    return np.log(M_piv)+(logRich-logL0-B_l*np.log((1+z)/1.3))/A_l
#TODO Make this one synonymous with it non-inverse version
#TODO Consider checking if logRich is a vector or a number, and acting accodingly.
sigma_mass = .1 #TODO idk what this should be; i suppose it's my discretion
df = 1

def invLogLamSample(logLam0, a, b, B_lam,sigma_mass, z,logRich, size = 100):
    #NOTE Returns ln(M), not log10(M)! This is how the formula is defined!
    mu = invLogLam(logLam0, a, b, B_lam, z, logRich)
    if sigma_mass == 0:
        return mu
    return np.array([t.rvs(df, loc = m, scale = sigma_mass, size =  size)\
                    for m in mu])#(logRich.shape[0], size)

#draw one set of samples, rather than re-drawing each cycle
#use truths as really really good guess. Can relax later.
logMassSamples = invLogLamSample(logL0_true, a_true, b_true, B_l_true,sigma_mass, redshift, logRichness, size = nSamples)
massSamples = np.exp(logMassSamples)

logPMass = log_n_approx(massSamples,redshift)
logPMass[massSamples<Mmin] = -np.inf

logPSample = np.array([t.logpdf(lms,df, loc = invLogLam(logL0_true, a_true, b_true, B_l_true, redshift, lr), scale = sigma_mass)\
                     for lms, lr in izip(logMassSamples, logRichness)])


def log_liklihood(logL0, a,b, B_l, sigma, z, logRich):

    logPRich = np.array([norm.logpdf(lr, loc =logLam(logL0, a, b, B_l, z, ms), scale = sigma)\
                         for lr, ms in izip(logRich, massSamples)])

    logL_k = logsumexp(logPRich+logPMass-logPSample, axis = 1) - np.log(nSamples)#mean of weights

    return np.sum(logL_k)

def log_posterior(theta,z, logRich):
    #print theta
    logL0,a,b, B_l, sigma = theta[:]
    b, B_l = b_true, B_l_true#no z information
    p = log_prior(logL0, a,b, B_l, sigma)
    if np.isfinite(p):
        p+=log_liklihood(logL0,a,b, B_l, sigma, z, logRich)
    return p

#set up sampler
nDim = 5

logL0_a, logL0_b = 1, 1 # Gamma
a_mean, a_spread = 1, 1.5
b_mean, b_spread = 0, .5
B_l_mean, B_l_spread = 1, 1.5
sigma_a, sigma_b = 1, 1 #Gamma

pos0 = np.zeros((nWalkers, nDim))
for row in pos0:

    row[0] = gamma.rvs(logL0_a, scale = logL0_b)
    row[1] = norm.rvs(loc = a_mean, scale = a_spread)
    row[2] = norm.rvs(loc = b_mean, scale = b_spread)
    row[3] = norm.rvs(loc = B_l_mean, scale = B_l_mean)
    row[4] = gamma.rvs(sigma_a, scale = sigma_b)

sampler = mc.EnsembleSampler(nWalkers, nDim, log_posterior, args=[redshift, logRichness],threads = nCores)
nburn = int(nSteps)/5

sampler.run_mcmc(pos0, nSteps)

if nCores> 1:
    sampler.pool.terminate()#there's a bug in emcee that creates daemon threads. This kills them.

chain = sampler.chain[:,nburn:, :].reshape((-1, nDim))

del(sampler)

MAP = chain.mean(axis = 0)
print MAP
labels = ['logL0', 'a', 'b','B_l','sigma']
print '\tMCMC\tTrue'
for label, val, truth in zip(labels, MAP, [logL0_true, a_true, b_true, B_l_true, sigma_l_true]):
    print '%s:\t%.3f\t%.3f'%(label, val, truth)

np.savetxt('is_chain_%dw_%ds.gz'%(nWalkers, nSteps), chain)
