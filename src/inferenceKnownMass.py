#!/u/ki/swmclau2/PYENV/bin/python
#@Author Sean McLaughlin
desc ='''
This module will eventually become part of the larger Pangloss structure.

I'm naively porting my work from jupyter notebooks to a .py file, for easier running on large clusters. I hope to work on better code
structure in the future.

This module attempts to infer the richness mass relation for galaxy cluster. This one actually works with a much simpler version of the problem
where the masses of the clusters are known. This is diagnose bugs in my inference scheme.
'''

from argparse import ArgumentParser
parser = ArgumentParser(description = desc)

parser.add_argument('nWalkers', metavar = 'nWalkers', type = int, help =\
                    'Number of walkers to use in the MCMC sampler. Appropriate values are around 1000.')

parser.add_argument('nSteps', metavar = 'nSteps', type = int, help =\
                    'Number of steps for the sampler to take. Appropriate values are 100-500')

parser.add_argument('nCores', metavar = 'nCores', type = int, help=\
                    'Number of cores to allow the sampler to use. Limited by machine maximum.')

parser.add_argument('--noDisplay', dest = 'noDisplay', action = 'store_true', help =\
                    'If used, will assume the job is running on a machine with no display and will make adjustments.')

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
from scipy.stats import linregress, gamma, expon, lognorm
import emcee as mc
if args.noDisplay:
    import matplotlib
    matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns
from corner import corner
sns.set()

#dataDir = '/home/sean/Data/BuzzardSims/'
dataDir = '/nfs/slac/g/ki/ki19/des/erykoff/clusters/mocks/Buzzard/buzzard-1.1/des_y5/redmapper_v6.4.7/halos/'
hdulist = fits.open(dataDir+'buzzard-v1.1-y5_run_00340_lambda_chisq.fit')
data = hdulist[1].data

mass = data['M200']
rich = data['LAMBDA_CHISQ']
mass = mass[rich>0]
rich = rich[rich>0]#slice off null values

logMass = np.log10(mass)
logRich = np.log10(rich)

offset = 13.5

b, logA, r, p, err = linregress(logMass-offset, logRich)

def log_prior(a,b,sigma):

    if any(x<0 for x in (a,sigma)):
    #if sigma<0:
        return - np.inf
    t = np.arctan(b)
    #if t<0 or t>np.pi/2:
    if t<-np.pi/2 or t>np.pi/2:
        return -np.inf

    #Hyperparameters
    lambda_a = 1.0
    sigma_a, sigma_b = 1,1

    p = 0
    #Exponential in log a
    #p+= np.log(lambda_a)-lambda_a*np.log(a)
    #p+= np.log(lambda_a)-lambda_a*a #changed a => logA TODO Change variable name?
    p+=expon.logpdf(np.log(a), scale = 1/lambda_a)
    #Uniform in arctan(b)
    p+=np.log(2/np.pi)
    #Inv Gamma for sigma
    p-= gamma.logpdf(sigma,sigma_a, scale = sigma_b)
    return p

def log_liklihood(rich, M, a,b,sigma):
    p = 0

    #p-= np.sum(((b*np.log(M)+np.log(a)-np.log(rich))**2)/(2*sigma**2)+np.log(sigma*rich))
    #redefine A to be intercept at center rather than 0
    #p-= np.sum(((b*(np.log(M)-13.5)+np.log(a)-np.log(rich))**2)/(2*sigma**2)+np.log(sigma*rich))
    #p-= np.sum(((b*np.log(M)+a-np.log(rich))**2)/(2*sigma**2)+np.log(sigma*rich))#See Above
    #p-= np.sum(((b*np.log(M-13.5)+a-np.log(rich))**2)/(2*sigma**2)+np.log(sigma*rich))#See Above

    p+= np.sum(lognorm.logpdf(rich, sigma, loc = (b*(np.log(M)-offset)+np.log(a))))

    return p

def log_posterior(theta,rich, M):
    #print theta
    a,b,sigma = theta[:]
    p = log_prior(a,b,sigma)
    if np.isfinite(p):
        p+=log_liklihood(rich,M,a,b,sigma)
    #print '-'*50
    return p

nDim = 3

#a_log_mean, a_log_spread = -5, 2
a_mean, a_spread = 4, 1.5
b_mean, b_spread = .5, .25
sigma_mean, sigma_spread = 1, .5

pos0 = np.zeros((nWalkers, nDim))
for row in pos0:
    #a,b,sigma,m
    #a_try = -1
    #while a_try < 0:
        #a_try = 10**(a_log_mean+np.random.randn()*a_log_spread)
    #    a_try = a_mean+np.random.randn()*a_spread
    #row[0] = a_try
    row[0] = a_mean+np.random.randn()*a_spread
    row[1] = b_mean+np.random.randn()*b_spread
    sig_try = -1
    while sig_try < 0:
        sig_try = sigma_mean+np.random.randn()*sigma_spread
    row[2] = sig_try

sampler = mc.EnsembleSampler(nWalkers, nDim, log_posterior, args=[rich, mass],threads = nCores)
nBurn = int(nSteps/10)

np.random.seed(0)#"random"
sampler.run_mcmc(pos0, nSteps)

chain = sampler.chain[:,nBurn:, :].reshape((-1,nDim))

sampler.pool.terminate()#there's a bug in emcee that creates daemon threads. This kills them.
del(sampler)

MAP = chain.mean(axis = 0)
print 'MAP'
print MAP[:-1]
print 'OLS'
print logA, b

titles = ['$a$', '$b$', '$\sigma$']
sigma_true = 1 #just a guess so this will plot
corner(chain, labels = titles , truths = [logA, b, sigma_true])
if args.noDisplay:
    plt.savefig('corner.png')
else:
    plt.show()

plt.scatter(logMass, logRich, alpha = .01)
plt.plot(logMass, MAP[1]*logMass+MAP[0], label = 'MCMC')
plt.plot(logMass, b*logMass+logA, label = 'OLS')
plt.legend(loc= 'best')
if args.noDisplay:
    plt.savefig('scatter.png')
else:
    plt.show()

np.savetxt('chain_%dw_%ds.gz'%(nWalkers, nSteps), chain)
