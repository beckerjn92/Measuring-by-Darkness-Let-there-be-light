# This script corresponds to the (micro-) publication "Measuring by Darkness? Let there be light!" by Rainer Heintzmann & Jan Becker
# It visualizes the two inferring strategies 'nulling' and '(optical) amplification'.

# %% Import typical python-packages which are used
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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
    calculates the probability of measuring k events given an expectation value of pred
    :param k: number of events
    :param pred: expectation value of poisson distribution
    return: the poisson probability
    """ 
    if np.size(k)>1:
        pred = np.repeat(pred[:, :, np.newaxis], np.size(k), axis=2)
    
    P = np.exp(-pred)*(pred**k)/scipy.special.factorial(k)
    return P

# %% Let's start the inference process

# Define photon budget and get the unkown value of x
budget = 10
oracle = game.Oracle(budget)

# Close old and open new figure
plt.close('all')
fig = plt.figure(1,figsize=(16,8))
fig.canvas.manager.window.activateWindow()
fig.canvas.manager.window.raise_()

# Initialize some variables needed for the estimation
a = 1.0                # cost per iteration
mode = "nulling"            # nulling or amplify
RefPhotons = 10              # reference photons
sz = [128,128]              # size of window

# First subplot is for showing the unit circle, the probability map and the estimated and true x-value
fig.add_subplot(1,2,1)
plt.scatter((oracle._x.real+1.0)*sz[0]/2,(oracle._x.imag+1.0)*sz[1]/2, marker='x',c='r')

# Initialize the number of detected photons and create the complex plane
Detected = 0
xmap = 2.0 * xx(sz) + 1j * 2.0 * yy(sz)

# Initialize the maximum number of photons and create a ramp of all possible photons numbers
maxPhoton = RefPhotons + np.sqrt(RefPhotons)*5 # 5 sigma
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
usedBudget = 0

# Go through all iterations, trying to improve the guess.
for N in range(NIter):
    # Predict your current guess
    pred = predict(a, b, xmap)  # |a*x + b|²

    # Ask the oracle and update the used budget
    M = oracle.ask(a, b)
    usedBudget += a**2
    #print("Photons "+str(M))

    # Have you gone over the budget?!?
    if usedBudget > budget:
        raise ValueError("budget was used up too early")

    # Update the detected photons
    Detected += M

    # Obtain probability of current measurement
    P_this_measurement = Poisson(M[0], pred)
    if M>maxPhoton:
        print("\nWarning! MaxPhoton: "+str(maxPhoton)+", but the measurement was "+str(M)+"\n")
    probMap *=  P_this_measurement
    probMap /= np.sum(probMap)

    # Make copy which we'll use for displaying the result
    probMap2Show = np.copy(probMap)
    probMap2Show[probMap2Show>0] = probMap2Show[probMap2Show>0] + 2*np.mean(probMap2Show)

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

# Plot the unit circle plot
    plt.ion()
    fig.add_subplot(1,2,1)
    plt.scatter(np.floor((myX.real+1.0)*sz[0]/2),np.floor((myX.imag+1.0)*sz[1]/2),marker='o',c='tab:orange' ,s=1.0)
    plt.imshow(probMap2Show,cmap = cm.get_cmap('Blues'))
    plt.axis('off')
    plt.gca().invert_yaxis()

    # Plot the graph with the statistical measures
    fig.add_subplot(1,2,2)
    p1=plt.scatter(N,np.real(meanX)-np.real(oracle._x[0]),marker='o',c='b',s=10,label='meanXr resid')
    p2=plt.scatter(N,np.real(bestX)-np.real(oracle._x[0]),marker='o',c='r',s=10,label='bestXr resid')
    plt.scatter(N,np.imag(meanX)-np.imag(oracle._x[0]),marker='o',c='g',s=10,label='meanXi resid')
    plt.scatter(N,np.imag(bestX)-np.imag(oracle._x[0]),marker='o',c='y',s=10,label='bestXr resid')
    plt.scatter(N,np.array(Detected)/100.0,marker='o',c='m',s=10,label='detected')
    plt.scatter(N,StdX,marker='o',c='c',s=10,label='StdDev Real')
    plt.scatter(N,StdY,marker='o',c='k',s=10,label='StdDev Imag')
    plt.xlabel('Iterations',size=18)
    plt.ylabel('Estimate',size=18)
    if N==0:
        plt.legend(loc='upper right',bbox_to_anchor=(1.25, 1))
        
    #plt.savefig("Images/image_{:01d}.png".format(N))   
    plt.pause(0.01)

fig.add_subplot(1,2,1)
plt.scatter(np.floor((myX.real+1.0)*sz[0]/2),np.floor((myX.imag+1.0)*sz[1]/2),marker='x',c='tab:green')

#plt.savefig("Images/image_{:01d}.png".format(N+1))
#plt.pause(0.01)