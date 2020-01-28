# This script corresponds to the (micro-) publication "Measuring by Darkness? Let there be light!" by Rainer Heintzmann & Jan Becker
# It compares the two inferring strategies 'nulling' and '(optical) amplification' and shows that latter is superior in terms of  the uncertainty
# of the inferred guess and the fact that you'll in any case get an estimate.

# %% Import typical python-packages which are used
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import special
import game

# %% Define some function which will be used

def xx(mysize=(256,256)):
    '''
    Two-dimensional map of x-coordinates, which will vary between -0.5 to 0.5.
    :param mysize: number of pixels in each dimension
    :return: x-coordinates map
    '''
    sizeX, sizeY = mysize
    x = np.linspace(-0.5,0.5,sizeX)
    y = np.linspace(-0.5,0.5,sizeY)
    xx, yy = np.meshgrid(x,y)
    return xx

def yy(mysize=(256,256)):
    '''
    Two-dimensional map of y-coordinates, which will vary between -0.5 to 0.5.
    :param mysize: number of pixels in each dimension
    :return: y-coordinates map
    '''
    sizeX, sizeY = mysize
    x = np.linspace(-0.5,0.5,sizeX)
    y = np.linspace(-0.5,0.5,sizeY)
    xx, yy = np.meshgrid(x,y)
    return yy

def rr(mysize=(256,256)):
    '''
    Two-dimensional map of radial coordinates, which will vary between -sqrt(0.5) to sqrt(0.5).
    :param mysize: number of pixels in each dimension
    :return: radial-coordinates map
    '''
    x = xx(mysize)
    y = yy(mysize)
    r = np.sqrt(x**2 + y**2) 
    return r

def ramp1D(mysize):
    '''
    Creates a 1-dimensional ramp.
    :param mysize: size of ramp
    :return: the ramp itself
    '''
    ramp = np.linspace(0,mysize)
    ramp = np.broadcast_to(ramp,(1,1,np.size(ramp)))
    return ramp
    
def abssqr(img):
    '''
    Calculates the absolute squared of the input: y = |x|²
    :param img: input which should be abs. squared
    :return: abs. squared result
    '''
    return np.real(img*np.conjugate(img))

def predict(a, b, x):
    '''
    Obtain the prediction or measurement given the values of a,b and x, according to: y = |a*x + b|²
    :param a: complex number
    :param b: complex number
    :param x: complex number
    :return: prediction of our measurement
    '''
    res = abssqr(a*x+b)
    return res

def Poisson(k, pred):
    """ 
    Calculates the probability of measuring 'k' events given an expectation value of 'pred'
    :param k: number of events
    :param pred: expectation value of poisson distribution
    return: the poisson probability
    """
    return np.exp(-pred)*(pred**k)/scipy.special.factorial(k)



# %% Algorithm which tries to estimate x!

def findXVal(budget, oracle, a = 0.5, mode="amplify", RefPhotons=6, sz=[256,256], usePrior=True, chooseBestDir = True, doGainEstimate = False):
    """
    Algorithm estimating the correct x-value given the budget.
    :param budget: total budget of photons available
    :param oracle: oracle knowing the true value
    :param a: constant choice of parameter a
    :param mode: either "nulling" or "amplify"
    :param RefPhotons: reference number of photons
    :param sz: size of simulation
    :param usePrior: boolean variable
    :param: chooseBestDir: boolean variable
    :param: doGainEstimate: boolean variable
    """

    # Initialize the number of detected photons and create the complex plane
    Detected = 0
    xmap = 2.0 * xx(sz) + 1j * 2.0 * yy(sz)

    # Initialize the maximum number of photons and create a ramp of all possible photons numbers
    maxPhoton = RefPhotons + np.sqrt(RefPhotons)*5      # 5 sigma
    k_ramp = ramp1D(maxPhoton)

    # Initialize probability map
    probMap = abssqr(xmap) < 1.0
    probMap = probMap / np.sum(probMap)

    # Depending on inferring mode initialize value of b
    if mode == "nulling":
        b = 0.0
    elif mode == "amplify":
        b = np.sqrt(RefPhotons)
    else:
        raise ValueError("unknown strategy mode. Only nulling and amplify are allowed.")

    # Initialize number of possible iterations and set used budget to 0
    NIter = int(budget // (a**2))
    usedBudget=0

    # Go through all iterations, trying to improve the guess.
    for N in range(NIter):
        # Predict your current guess
        pred = predict(a, b, xmap)

        # Ask the oracle and update the used budget
        M = oracle.ask(a,b)
        usedBudget += a**2
        #print("Photons "+str(M))

        # Have you gone over the budget?!?
        if usedBudget > budget:
            raise ValueError("budget was used up too early")

        # Update the detected photons
        Detected += Detected    # M?

        # Obtain probability of current measurement
        P_this_measurement = Poisson(M[0], pred)
        if M>maxPhoton:
            print("\nWarning! MaxPhoton: "+str(maxPhoton)+", but the measurement was "+str(M)+"\n")
        probMap *=  P_this_measurement
        probMap /= np.sum(probMap)      # normalization!

        # Get expectation value
        mX = np.sum(xmap * probMap)
        meanX = mX

        # Obtain variance's and cross-variance
        diffMap = xmap-mX
        varX = np.sum(np.square(np.real(diffMap)) * probMap)
        varY = np.sum(np.square(np.imag(diffMap)) * probMap)
        varXY = np.sum(np.real(diffMap)*np.imag(diffMap) * probMap)
        StdX = np.sqrt(varX)
        StdY = np.sqrt(varY)

        # Find value with highest probability --> best estimate
        tmp = np.unravel_index(probMap.argmax(), probMap.shape)
        myX = xmap[tmp[0], tmp[1]]  # this is the current maximum likelihood estimate
        bestX = myX

        # Get gradient of probability map and find steepest direction
        (gx,gy) = np.gradient(probMap)
        gradSqr = gx**2+gy**2
        tmp = np.unravel_index(gradSqr.argmax(), gradSqr.shape)
        maxGrad = xmap[tmp[0], tmp[1]]

        # Depending on measurement scheme adjust b-value
        if mode == "nulling":
            b = - a*bestX # try to cancel destructively
            phi = np.random.random()*2.0*np.pi
        elif mode == "amplify":
            phi = np.pi/2*N

            # New value for b!
            b = np.sqrt(RefPhotons) * np.exp(-1j*phi)
        else:
            raise ValueError("unknown strategy mode. Only nulling and amplify are allowed.")

    if oracle.ask(a,0) is not None:
        print("The budget was not used up")
        raise ValueError("The budget was not used up")

    return (myX, varX, varY, varXY)

# %% Function which calls 'FindXVal' several times

def MultiTest(budget, NumStat, NumTries, mode="amplify", chooseBestDir=True, usePrior=True, a=0.5, RefPhotons=6):
    """
    This function
    :param budget: total budged of photons available
    :param NumStat:
    :param NumTries:
    :param 
    """

    # Initialize some variables
    sumRes = 0 + 0j
    sumRes2r = 0; sumRes2i = 0; allStdDevX = 0; allStdDevY = 0
    sumAlgVarX=0; sumAlgVarY=0; sumAlgVarXY=0

    # Let's try to do the estimation for 'NumStat' different x-values
    for n in range(NumStat):
        # Initialize some variables
        sumPos = 0 + 0j; sumPos2r = 0; sumPos2i = 0

        # Ask the oracle for a new (unkown) x-value
        oracle = game.Oracle(budget)

        # For one randomly assigned true x-value, we try to infer the value 'NumTries'-times
        for t in range(NumTries):
            oracle.total_cost = 0  # reset the oracle for the same _x
            oracle.game_over = False
            #print("Oracle no. " + str(n) + ", Test no. " + str(t) + ", oracle: " + str(oracle._x))
            #print("----------------------------------------------------")

            # Try to get a good estimate of the unkown x-value
            (bX, varX, varY, varXY) = findXVal(budget, oracle, a=a, RefPhotons=RefPhotons, mode=mode, chooseBestDir=chooseBestDir, usePrior=usePrior)

            # Obtain some statistical values
            resid = bX - oracle._x[0]       # residiuals in x- & y-directon
            sumRes += resid                 # sum of residuals
            sumAlgVarX +=varX; sumAlgVarY += varY; sumAlgVarXY += varXY         # sum of the algorithm's variance's
            sumRes2r += np.real(resid) ** 2; sumRes2i += np.imag(resid) ** 2    # residual along real & imaginary axis
            sumPos += bX    # sum of estimated position
            sumPos2r += np.real(bX) ** 2;sumPos2i += np.imag(bX) ** 2       # sum of estimated positin along real & imaginary axis

        # Obtain standard deviation of estimation in 'x' & 'y'-direction
        StdDevX = np.sqrt(sumPos2r / NumTries - np.real(sumPos / NumTries) ** 2)
        StdDevY = np.sqrt(sumPos2i / NumTries - np.imag(sumPos / NumTries) ** 2)
        #print("\nStdDevX: " + str(StdDevX))
        #print("StdDevY: " + str(StdDevY))
        #print("----------------------------------------------------")
        #print("----------------------------------------------------")
        allStdDevX += StdDevX
        allStdDevY += StdDevY

        # Output is normalized!
    return (sumRes/NumStat/NumTries, np.sqrt(sumRes2r/NumStat/NumTries),np.sqrt(sumRes2i/NumStat/NumTries),allStdDevX/NumTries,allStdDevY/NumTries, np.sqrt(sumAlgVarX/NumStat/NumTries), np.sqrt(sumAlgVarY/NumStat/NumTries), np.sqrt(np.abs(sumAlgVarXY)/NumStat/NumTries))


# %% Define which inferring scheme you want to use and how big our budget is!

# The budget we allow in the simulation
mybudget = 10

# Let's do the numerical experiment with the 'nulling'-scheme
res = MultiTest(mybudget, 10, 10, mode="nulling")       # 'nulling' or 'amplify'

# Print the results
(Bias, TrueStdX, TrueStdY, StdDevX, StdDevY, AlgStdX, AlgStdY, AlgStdXY) = res
print(" NULLING: ")
print("\n Bias: "+str(Bias))
print(" StdDev(Res X-TrueX): "+str(TrueStdX))
print(" StdDev(Res Y-TrueY): "+str(TrueStdY))
print(" Mean of StdDevX: " + str(StdDevX))
print(" Mean of StdDevY: " + str(StdDevY))
print(" Mean of Algo StdDev X: " + str(AlgStdX))
print(" Mean of Algo StdDev Y: " + str(AlgStdY))
print(" Mean of Algo StdDev XY: " + str(AlgStdXY))



# Let's do the numerical experiment with the 'amplification'-scheme
res = MultiTest(mybudget, 10, 10, mode="amplify")       # 'nulling' or 'amplify'

# Print the results
(Bias, TrueStdX, TrueStdY, StdDevX, StdDevY, AlgStdX, AlgStdY, AlgStdXY) = res
print(" AMPLIFICATION: ")
print("\n Bias: "+str(Bias))
print(" StdDev(Res X-TrueX): "+str(TrueStdX))
print(" StdDev(Res Y-TrueY): "+str(TrueStdY))
print(" Mean of StdDevX: " + str(StdDevX))
print(" Mean of StdDevY: " + str(StdDevY))
print(" Mean of Algo StdDev X: " + str(AlgStdX))
print(" Mean of Algo StdDev Y: " + str(AlgStdY))
print(" Mean of Algo StdDev XY: " + str(AlgStdXY))
