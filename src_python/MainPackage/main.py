import numpy as np
import pandas as pd
from .main_helpers import *
from scipy.stats import mode

class MixtureModelBernoulli():

    """
    Class Constructor
    ---------------------
    num_classes: int
        The number of classes you want to fit the data to
    random_state: int (optional)
        Determines the seed for the random number generator
    max_iter: int
        The max number of iterations done in the gibbs sampling
    burn_in: int
        Amount of initial samples excluded from the MCMC chain
        Note: burn_in < max_iter
    """
    def __init__(self,num_classes=1,random_state=None,max_iter=500,burn_in=100):
        self.num_classes = num_classes
        self.max_iter = max_iter
        self.burn_in = burn_in
        # Boolean to see if model was fitted
        self.fitted_model = False
        # Set up random number generator. This is used to sample values
        self.rand_num_gen = np.random.default_rng(random_state)



    # Functions used to train the model

    """
    fit the model to the given data X
    ------------------------
    X : array-like of shape (num_samples, num_params)
    """
    def fit(self,X):
        # Filter and check the input
        X = check_input_X(X)
        self.num_samples = X.shape[0]
        self.num_params = X.shape[1]

        # Initialize Parameters
        # todo get rid of these inputs?
        self.initialize_params(self.num_samples,self.num_params)

        # Initialize lists for samples
        # todo use np arrays of fixed size instead of lists
        self.samples_theta_params = list()
        self.samples_class_assignments = list()
        self.samples_class_probabilities = list()
        self.samples_class_membership_scores = list()

        # HYPER PARAMETERS
        # TODO WHERE TO PUT THIS?
        theta_hyperparams = np.array([2,2])
        class_prob_hyperparams = 2*np.ones_like(self.class_probabilities)


        # other stuff
        ohe_converter = np.arange(self.num_classes)

        # Main Loop
        num_iter = 1
        while num_iter < (self.max_iter+self.burn_in):

            # Calculate num_members_per_class
            num_members_per_class = np.bincount(self.class_assignments,minlength=self.num_classes)


            # Calculate num_correct_per_class_by_q
            # todo double check this
            num_correct_per_class_by_q = np.zeros((self.num_classes,self.num_params))
            for i in range(self.num_samples):
                num_correct_per_class_by_q[self.class_assignments[i]] += X[i]


            # Sample theta_params
            for c in range(self.num_classes):
                for j in range(self.num_params):
                    alpha = theta_hyperparams[0]+num_correct_per_class_by_q[c,j]
                    beta = theta_hyperparams[1]+num_members_per_class[c]-num_correct_per_class_by_q[c,j] # todo double check
                    self.theta_params[c,j] = self.rand_num_gen.beta(alpha,beta)
            # Append to samples list
            if num_iter>self.burn_in:
                self.samples_theta_params.append(self.theta_params.copy())


            # Calculate class_membership_scores
            self.class_membership_scores = np.ones_like(self.class_membership_scores)
            for i in range(self.num_samples):
                for c in range(self.num_classes):
                    self.class_membership_scores[i, c] = np.prod(self.theta_params[c,:]**X[i,:] * (1-self.theta_params[c,:])**(1-X[i,:])) * self.class_probabilities[c]
            # Normalize
            self.class_membership_scores = self.class_membership_scores / np.sum(self.class_membership_scores,axis=1,keepdims=True)
            # Append to samples list
            if num_iter>self.burn_in:
                self.samples_class_membership_scores.append(self.class_membership_scores.copy())


            # Sample class_assignments
            # todo can be optimized
            for i in range(self.num_samples):
                one_hot_encoding = self.rand_num_gen.multinomial(n=1,pvals=self.class_membership_scores[i])
                # Convert one hot encoding to integer encoding to get a single integer that represents the class assignement
                self.class_assignments[i] = np.dot(one_hot_encoding,ohe_converter)
            # Append to samples list
            if num_iter>self.burn_in:
                self.samples_class_assignments.append(self.class_assignments.copy())


            # Sample class_probabilities
            self.class_probabilities = self.rand_num_gen.dirichlet(class_prob_hyperparams+num_members_per_class)
            # Append to samples list
            if num_iter>self.burn_in:
                self.samples_class_probabilities.append(self.class_probabilities.copy())

            # End of loop
            # todo how should we implement tolerance?
            # Make it so we save after every iteration in order to do inference.
            # See convergence
            # Use mean of samples in order to predict.
            # After we have clustered how can we cluster one more sample?
            num_iter +=1

        # DONE!
        self.fitted_model = True

        # Convert to numpy
        self.samples_theta_params = np.array(self.samples_theta_params)
        self.samples_class_assignments = np.array(self.samples_class_assignments)
        self.samples_class_probabilities = np.array(self.samples_class_probabilities)
        self.samples_class_membership_scores = np.array(self.samples_class_membership_scores)

    """
    fit the model to the given data X and return the class assignments
    ------------------------
    X : array-like of shape (num_samples, num_params)
    -----------------------
    new_class_assignments: np array with length (num_samples)
        The predicted assignments for the training data
    """
    def fit_predict(self,X):
        self.fit(X)
        return self.get_params()[2]


    # Main Methods

    """
    returns the predicted parameters of the model by computing the average of the samples
    ----------------
    ----------------
    mean_theta_params: np array with shape (num_classes,num_params)
        The predicted theta matrix
    mean_class_probabilities: np array with length (num_classes)
        The predicted prior distribution
    mean_class_assignments: np array with length (num_samples)
        The predicted assignments for the training data
    """
    def get_params(self):
        if not self.fitted_model:
            raise ValueError("You must fit the model before asking for the parameters")

        # Find means
        self.mean_theta_params = np.average(self.samples_theta_params,axis=0)
        self.mean_class_probabilities = np.average(self.samples_class_probabilities,axis=0)
        self.mean_class_assignments = mode(self.samples_class_assignments,axis=0)[0][0]

        return self.mean_theta_params, self.mean_class_probabilities,self.mean_class_assignments

    """
    Find the fuzzy partition matrix for the training data
    ----------------
    ----------------
    class_membership_scores: np array with length (num_samples,num_classes)
        The probability that each person belongs to a specific class
    """
    def get_class_membership_scores(self):
        if not self.fitted_model:
            raise ValueError("You must fit the model before asking for the parameters")

        # Find means
        self.mean_class_membership_scores = np.average(self.samples_class_membership_scores,axis=0)
        return self.class_membership_scores

    """
    Generates new data of size N from the fitted parameters
    ----------------
    num_samples: int (optional)
        the number of new samples you want. Default will use the number of samples from fitted data
    random_state: int (optional)
        Determines the seed for the random number generator
    ----------------
    new_X: np array with shape (num_samples,n_features)
        The new data matrix. num_features is the number of features in the fitted data
    new_class_assignments: np array with length (num_samples)
        The assignments for the newly generated data
    """
    def resample(self, num_samples=None,random_state=None):
        if not self.fitted_model:
            raise ValueError("You must fit the model before resampling")
        elif num_samples is None:
            num_samples = self.num_samples
        elif random_state is not None:
            # Reset the seed if given
            self.rand_num_gen = np.random.default_rng(random_state)
        # Sample new class_assignments using prior
        # todo can be optimized
        new_class_assignments = self.rand_num_gen.multinomial(n=1, size=(num_samples), pvals=self.class_probabilities)
        # Convert one hot encoding to integer encoding to get a single integer that represents the class assignement
        new_class_assignments = np.dot(new_class_assignments,np.array([0, 1, 2, 3]))

        # Sample answers from for each student
        # todo this might be able to be optimized
        new_X = np.zeros((num_samples,self.num_params),dtype=bool)
        for i in range(num_samples):
            new_X[i] = self.rand_num_gen.binomial(1,self.theta_params[new_class_assignments[i]])
        return new_X, new_class_assignments

    # Functions not typically used by user

    """
    Initializes all the parameters.
    ----------------------
    num_samples: int (optional)
        the number of new samples in the data. Default will use the number of samples from fitted data
    num_params: int (optional)
        the number of parameter in the data. Default will use the number of samples from fitted data
    """
    def initialize_params(self,n_samples,num_params):
        self.theta_params = self.rand_num_gen.uniform(size=(self.num_classes,num_params))
        self.class_assignments = self.rand_num_gen.integers(0,self.num_classes,size=(n_samples))
        self.class_membership_scores = np.ones((n_samples,self.num_classes))
        self.class_probabilities = self.rand_num_gen.uniform(size=(self.num_classes))
        self.class_probabilities = self.class_probabilities / np.sum(self.class_probabilities)