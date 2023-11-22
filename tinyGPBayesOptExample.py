# Import required packages
import itertools
import jax.numpy as jnp
import jax.scipy as jsp
import jaxopt
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import numpy as np
import tinygp

# Variables:
# theta = hyperparameters for covariance function [theta[0] is the noise]
# D_n = current data set (observed x and corresponding y points) D_n[:,0] = observed x coordinates, D_n[:,1] = observed y coordinates
# x_q = query x point(s)

# DIFFERENT OBJECTIVE FUNCTIONS TO TEST. REMOVE THE NUMBER FROM THE DESIRED OBJECTIVE FUNCTION SUCH THAT IT IS INPUT IN THE CODE.

# ACKLEY OBJECTIVE FUNCTION WITH O.O1 NORMALLY DISTRIBUTED NOISE.
# SINCE THE ACKLEY FUNCTION CAN BE EXTENDED TO INFINITE DIMENSIONS, NO NEED TO CHANGE ANYTHING HERE.
def Objective_Function(x):
    # Model of the Objective Function. Returns noisy data readings.
    d = len(x)
    a = 20
    b = 0.2
    c = 2*np.pi
    sum1 = 0
    sum2 = 0
    for i in range(0,d):
      sum1 = sum1 + np.square(x[i])
      sum2 = sum2 + np.cos(c*x[i])
    term1 = -a * np.exp(-b*np.sqrt(sum1))
    term2 = -np.exp(sum2)
    phi = term1 + term2 + a + np.exp(1)
    # Add normally distributed noise
    n = np.random.normal(loc=0.0, scale=0.00, size=1)
    y = phi + n
    return y

# GRAMACY & LEE OBJECTIVE FUNCTION WITH O.O1 NORMALLY DISTRIBUTED NOISE.
def Objective_Function2(x):
    # Model of the Objective Function. Returns noisy data readings.
    term1 = np.sin(10*np.pi*x) / (2*x)
    term2 = (x-1)**4
    phi = term1 + term2
    phi = -phi
    # Add normally distributed noise
    n = np.random.normal(loc=0.0, scale=0.0, size=1)
    y = phi + n
    return y

# 1 DIMENSIONAL DROP-WAVE OBJECTIVE FUNCTION WITH O.O1 NORMALLY DISTRIBUTED NOISE.
def Objective_Function3(x):
    # Model of the Objective Function. Returns noisy data readings.
    x1 = x;
    x2 = 0;

    frac1 = 1 + jnp.cos(12*jnp.sqrt(x1**2+x2**2));
    frac2 = 0.5*(x1**2+x2**2) + 2;

    phi = -frac1/frac2;
    # Add normally distributed noise
    n = np.random.normal(loc=0.0, scale=0.00, size=1)
    y = phi + n
    return y

# INITIAL DATAPOINTS

# generate random [1] dimensional vector(s) in the desired range that act as the initially observed input datapoints.
num_vectors = 5

# WE HAVE CHANGED THE DIMENSION TO 1 [COMPARED TO "Bayesian_Optimisation_Lib_V8.ipynb"]
dim = 1
x_o = np.random.uniform(-5, 5, size=(num_vectors, dim))

# Use the Objective Function ('Black Box') to generate the corresponding y values (with noise)
y_o = list()
for i in range(0,len(x_o)):
  x_i = x_o[i,:]
  y_i = Objective_Function(x_i)
  y_o.append(y_i[0])

# Concatenate the data points (list of input datapoints, x_o, and corresponding output datapoints, y_o)
y_o = np.asarray(y_o).reshape(-1,1)
D_o = np.hstack((x_o,y_o))

# Plot a scatter graph of the data
plt.plot(D_o[:,0],D_o[:,-1],'o')
plt.xlabel('x')
plt.ylabel('y')

# DEFINE A DICTIONARY OF THE HYPERPARAMETERS. FOR INITIAL SIMIPLICITY, THETA WILL HAVE A SET STRUCTURE OF AMPLITUDES, SCALES ETC. OF DIFFERENT COMPONENTS OF A COMPLEX COVARIANCE FUNCTION. EACH COMPONENT CAN BE SET TO 0 OR 1 BY FUNCTION 'CONSTRUCT_KERNEL_STRUCTURE':
# Define initial variables (initial guess of hyper-parameters):
initial_theta = {
    "mean": np.float64(340.0),
    "log_diag": np.log(0.01),
    "log_amps": np.float64(0.0),
    "log_scales": np.zeros(1), # NB: WE NOW HAVE THE NUMBER OF PARAMETERS IN LOG SCALES.
}

def anthropic_AI(source_code):
  # Function to call Claude and determine/search for the structure of the covariance function. Returns a list of binary values [ones or zeros, to switch different components of the kernel structure on or off].
  # MAYBE BETTER TO INITIALISE THE HYPERPARAMETERS HERE?
  # EXTENSION: INCLUDE BAYESIAN OPTIMISATION IN MODEL SPACE HERE?
  # At the moment, we are simply adding different kernel structures, but look into different ways of combining e.g. multiplication.
  # before integrating with anthropic AI, we stick with a pre-defined switch structure [here the 'source code is = [1, 1, 1, 1, 0]'].
  switch = source_code
  return switch

def construct_final_kernel_structure(theta, switch):
# theta["log_scales"] refers to an array!
  # Function to construct the final kernel structure. The Parameters 'Theta' and 'Switch' turn a generic, additive kernel structure into a specific one:
    kernel = switch[0]*jnp.exp(theta["log_amps"]) * tinygp.transforms.Linear(
        jnp.exp(-theta["log_scales"]), tinygp.kernels.ExpSquared()
    )
    return kernel

# Create model for the objective function.
# FUTURE EXTENSION: INTEGRATE THE ANTHROPIC AI FUNCTION AND CREATING THE FINAL COVARIANCE FUNCTION INTO THE 'MODEL' CLASS.

class Model:

    def __init__(self, D_n, source_code):
        self.D_n = D_n
        self.x_n = D_n[:,:-1]
        self.y_n = D_n[:,-1]
        self.source_code = source_code
        self.switch = []

    def build_GPR(self, theta):
        kernel = construct_final_kernel_structure(theta, self.switch)
        # structure of inputs: tinygp.GaussianProcess(kernel = kernel, X = D_n, diag = jnp.exp(theta["log_diag"], noise = , mean = , solver=)
        GPR_model = tinygp.GaussianProcess(kernel = kernel, X = self.x_n, diag = jnp.exp(theta["log_diag"]), mean = np.mean(self.y_n))
        return GPR_model

    def negative_log_likelihood(self, theta):
        nll = -self.build_GPR(theta).log_probability(self.y_n)
        return nll

    def optimise_theta(self):
        nll_minimizer = jaxopt.ScipyMinimize(fun = self.negative_log_likelihood)
        optimised_theta = nll_minimizer.run(initial_theta).params
        return optimised_theta

    def f(self, optimised_theta=dict({})): # Final Surrogate Function [i.e. final Objective Function Model]
        self.switch = anthropic_AI(self.source_code)
        if len(optimised_theta) == 0:
            # Use Claude to determine what parts of Covariance Matrix are required/not required
            print('Optimising theta')
            optimised_theta = self.optimise_theta()
        else:
          print('Theta already optimised')
        GPR_model = self.build_GPR(optimised_theta)
        return GPR_model, optimised_theta

# TESTING WHETHER THE SURROGATE MODEL ACTUALLY WORKS
source_code = [1, 1, 1, 1, 0]
GPR_Model = Model(D_o, source_code)
Tuned_GPR_Model, optimised_theta = GPR_Model.f()

# Variable Structures
# OBSERVED DATA POINTS
# D_n = [[x1 x2 y]
#        []]
# x_n = [[x1 x2]
#         ]
# y_n = []

# source_code = [1, 1, ..., 0]

# Query Point
# x_q = [[x1_q x2_q]] Array with query points

# USING AN ACQUISITION FUNCTION TO SELECT THE NEXT INPUT. THE ABOVE MODEL IS USED WITHIN THE ACQUISITION FUNCTION.
# alpha : the expected improvement (utility)

class Noiseless_Expected_Improvement:

  def __init__(self, D_n, source_code, initial_x_q, optimised_theta=dict({})):
    self.D_n = D_n
    self.x_n = D_n[:,:-1]
    self.y_n = D_n[:,-1]
    self.source_code = source_code
    self.switch = []
    self.initial_x_q = initial_x_q

    # Create the Surrogate Model based on the observed data points D_n.
    GPR_Model = Model(D_n, source_code)
    self.Tuned_GPR_Model,_ = GPR_Model.f(optimised_theta)

  def alpha(self, x_q):
    phi_star = min(self.y_n)
    mu = self.Tuned_GPR_Model.condition(self.y_n, X_test = x_q).gp.loc
    sigma = jnp.sqrt(self.Tuned_GPR_Model.condition(y=self.y_n, X_test = x_q).gp.variance)
    # negative alpha (i.e. expected improvement i.e. utility) to minimise rather than maximise.
    nalpha = ((mu-phi_star) * jsp.stats.norm.cdf(phi_star, loc = mu, scale = sigma) - (sigma**2) * jsp.stats.norm.pdf(phi_star, loc = mu, scale = sigma)) # jax.scipy.stats.norm.cdf(x, loc=0 [mean], scale=1 [standard_deviation])[source]
    nalpha = nalpha[0] # Converting from an array to a single value
    return nalpha

  def optimise_alpha(self):
    nalpha_minimizer = jaxopt.ScipyMinimize(fun = self.alpha)
    next_x_q = nalpha_minimizer.run(self.initial_x_q).params # find the query point xq that will maximise the expected improvement.
    #next_x_q = jsp.optimize.basinhopping(fun = self.alpha, x0 = self.initial_x_q)
    next_x_q = next_x_q[0,:]
    return next_x_q

# We amend the Bayesian Optimisation function to plot the acquisition function and surrogate model at every step.
# We remove the 'optimised_theta' input, such that the hyperparameters are re-optimised on each step.
def Bayesian_Optimisation(D_o, source_code, initial_x_q, termination = 10, optimised_theta=dict({})):
  n = 0              # Counter
  D_n = D_o          # Set initial data set as current data set

  while n < termination:
    Acquisition_Function = Noiseless_Expected_Improvement(D_n, source_code, initial_x_q)

    # VISUALISE
    # Plot the current dataset
    fig, ax1 = plt.subplots()
    ax1.plot(D_o[:,0],D_o[:,1],'ok',label='Observed Data points')
    # Plot the surrogate model with uncertainty (standard deviation)
    surrogate_x = np.array([np.linspace(-5,5,100)]).T
    surrogate_y = Acquisition_Function.Tuned_GPR_Model.condition(Acquisition_Function.y_n, X_test = surrogate_x).gp.loc
    surrogate_sd = np.sqrt(Acquisition_Function.Tuned_GPR_Model.condition(Acquisition_Function.y_n, X_test = surrogate_x).gp.variance)
    ax1.plot(surrogate_x, surrogate_y,'-k', label = 'Surrogate Model')
    ax1.fill_between(surrogate_x.flatten(), surrogate_y + surrogate_sd, surrogate_y - surrogate_sd, color="C0", alpha=0.5)
    ax1.legend()
    # Plot the Acquisition Function
    ax2 = ax1.twinx()
    alpha_x = np.linspace(-5,5,500)
    alpha_y = [Acquisition_Function.alpha(np.array([[x_q]])) for x_q in alpha_x]
    ax2.plot(alpha_x, alpha_y, '-r',label='Acquisition Function')
    ax2.plot(alpha_x[np.argmin(alpha_y)],min(alpha_y), 'xr')
    ax2.legend()
    plt.show()
    print('Next x value from acquisition function vector:',alpha_x[np.argmin(alpha_y)])

    # Minimise the Acquisition Function
    x_q = Acquisition_Function.optimise_alpha()             # x_q should be an array [x1_q x2_q]
    y = Objective_Function(x_q)[0]                          # [0] so that y is a value, not an array
    print('Query Input:{} and Corresponding Output from Objective Function:{}'.format(x_q,y))

    # Update Data Set
    n = n+1                                                 # increment counter
    D_n = np.vstack((D_n,np.hstack((x_q,y))))               # Update data set i.e. D_n <- D_{n+1}
    initial_x_q = np.array([D_n[np.argmin(D_n[:,-1]),:-1]]) # Initialise x_q (optimisation) with the x array that corresponds to the smallest y value, since noiseless EI favours exploitation over exploration.
    print('Updated Data Set has shape: ',np.shape(D_n))

  return(D_n, initial_x_q)

# TESTING WHETHER BAYESIAN OPTIMISATION ACTUALLY WORKS. INITIALISE THESE VALUES ONCE, BUT THEN DO NOT REPEAT,
# SINCE OTHERWISE "initial_x_q" WILL BE RE-INITIALISED.

# Test the Bayesian Optimisation Function
initial_x_q = np.float64([[1.1]])
source_code = [1, 1, 1, 1, 0]

D_n, initial_x_q = Bayesian_Optimisation(D_o, source_code, initial_x_q, termination = 1)

print('Smallest y value',min(D_n[:,-1]))
print('X Coordinate of smallest y value:', D_n[np.argmin(D_n[:,-1]),:-1])

D_o = D_n