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
        self._max_iter = max_iter
        self._burn_in = burn_in
        # Boolean to see if model was fitted
        self._fitted_model = False
        # Set up random number generator. This is used to sample values
        self._rand_num_gen = np.random.default_rng(random_state)



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
        theta_params, class_probabilities, class_assignments = self.initialize_params(self.num_samples,self.num_params)

        # Initialize lists for samples
        # todo use np arrays of fixed size instead of lists
        self.samples_theta_params = np.zeros((self._max_iter,self.num_classes,self.num_params))
        self.samples_class_assignments = np.zeros((self._max_iter,self.num_samples))
        self.samples_class_probabilities = np.zeros((self._max_iter,self.num_classes))

        # HYPER PARAMETERS
        # TODO add the option to change these
        theta_hyperparams = np.array([2,2])
        class_prob_hyperparams = 2*np.ones_like(class_probabilities)


        # other stuff
        ohe_converter = np.arange(self.num_classes)

        # Main Loop
        num_iter = 1
        while num_iter < (self._max_iter+self._burn_in):

            # Calculate num_members_per_class
            num_members_per_class = np.bincount(class_assignments,minlength=self.num_classes)


            # Calculate num_correct_per_class_by_q
            num_correct_per_class_by_q = np.zeros((self.num_classes,self.num_params))
            for i in range(self.num_samples):
                num_correct_per_class_by_q[class_assignments[i]] += X[i]

            # Sample theta_params
            alpha  = theta_hyperparams[0] + num_correct_per_class_by_q
            beta = theta_hyperparams[1] + num_members_per_class[:,np.newaxis] - num_correct_per_class_by_q
            theta_params = self._rand_num_gen.beta(alpha, beta)
            # Append to samples list
            if num_iter>self._burn_in:
                self.samples_theta_params[num_iter-self._burn_in] = theta_params.copy()


            # Calculate class_membership_scores
            self.class_membership_scores = np.ones_like(self.class_membership_scores)
            for i in range(self.num_samples):
                for c in range(self.num_classes):
                    self.class_membership_scores[i, c] = np.prod(theta_params[c,:]**X[i,:] * (1-theta_params[c,:])**(1-X[i,:])) * class_probabilities[c]
            # Normalize
            self.class_membership_scores = self.class_membership_scores / np.sum(self.class_membership_scores,axis=1,keepdims=True)


            # Sample class_assignments
            for i in range(self.num_samples):
                one_hot_encoding = self._rand_num_gen.multinomial(n=1,pvals=self.class_membership_scores[i])
                # Convert one hot encoding to integer encoding to get a single integer that represents the class assignement
                class_assignments[i] = np.dot(one_hot_encoding,ohe_converter)
            # Append to samples list
            if num_iter>self._burn_in:
                self.samples_class_assignments[num_iter-self._burn_in] = class_assignments.copy()

            # Sample class_probabilities
            class_probabilities = self._rand_num_gen.dirichlet(class_prob_hyperparams+num_members_per_class)
            # Append to samples list
            if num_iter>self._burn_in:
                self.samples_class_probabilities[num_iter-self._burn_in] = class_probabilities.copy()

            # End of loop
            num_iter +=1

        # DONE!
        self._fitted_model = True
        self.X = X
        # Find means
        self.mean_theta_params = np.average(self.samples_theta_params,axis=0)
        self.mean_class_probabilities = np.average(self.samples_class_probabilities,axis=0)
        self.mean_class_assignments = mode(self.samples_class_assignments,axis=0)[0][0].astype(int)
        # Calculate class_membership_scores
        self.class_membership_scores = np.ones_like(self.class_membership_scores)
        for i in range(self.num_samples):
            for c in range(self.num_classes):
                self.class_membership_scores[i, c] = np.prod(self.mean_theta_params[c, :] ** X[i, :] * (1 - self.mean_theta_params[c, :]) ** (1 - X[i, :])) * self.mean_class_probabilities[c]
        # Normalize
        self.class_membership_scores = self.class_membership_scores / np.sum(self.class_membership_scores, axis=1,keepdims=True)
        

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
        if not self._fitted_model:
            raise ValueError("You must fit the model before asking for the parameters")

        return self.mean_theta_params, self.mean_class_probabilities,self.mean_class_assignments

    """
    Find the fuzzy partition matrix for the training data
    ----------------
    ----------------
    class_membership_scores: np array with length (num_samples,num_classes)
        The probability that each person belongs to a specific class
    """
    def get_class_membership_scores(self):
        if not self._fitted_model:
            raise ValueError("You must fit the model before asking for the parameters")
        try:
            return self.mean_class_membership_scores
        except:
            theta_params,class_probabilities,class_assignments = self.get_params()
            # Calculate class_membership_scores
            self.mean_class_membership_scores = np.ones_like(self.class_membership_scores)
            for i in range(self.num_samples):
                for c in range(self.num_classes):
                    self.mean_class_membership_scores[i, c] = np.prod(theta_params[c, :] ** self.X[i, :] * (1 - theta_params[c, :]) ** (1 - self.X[i, :])) * class_probabilities[c]
            # Normalize
            self.mean_class_membership_scores = self.class_membership_scores / np.sum(self.class_membership_scores,
                                                                                      axis=1,keepdims=True)
            return self.mean_class_membership_scores

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
        if not self._fitted_model:
            raise ValueError("You must fit the model before resampling")

        # Reset the seed if given
        if random_state is not None:
            # Reset the seed if given
            self._rand_num_gen = np.random.default_rng(random_state)

        if num_samples is None:
            num_samples = self.num_samples
            # Sample new class_assignments using class membership scores
            new_class_assignments = np.zeros(num_samples)
            ohe = np.arange(self.num_classes)
            for i in range(num_samples):
                new_class = self._rand_num_gen.multinomial(n=1,pvals=self.get_class_membership_scores()[i])
                # Convert one hot encoding to integer encoding to get a single integer that represents the class assignement
                new_class_assignments[i] = np.dot(new_class,ohe)
            new_class_assignments = np.round(new_class_assignments).astype(int)
        else:
            # Sample new class_assignments using prior
            new_class_assignments = self._rand_num_gen.multinomial(n=1,
                                                                   size=(num_samples),
                                                                   pvals=self.mean_class_probabilities)
            # Convert one hot encoding to integer encoding to get a single integer that represents the class assignement
            new_class_assignments = np.dot(new_class_assignments, np.arange(0, new_class_assignments.shape[1]))

        # Sample answers from for each student
        # todo this might be able to be optimized
        new_X = np.zeros((num_samples,self.num_params),dtype=bool)
        for i in range(num_samples):
            new_X[i] = self._rand_num_gen.binomial(1,self.mean_theta_params[new_class_assignments[i]])
        return new_X, new_class_assignments

    """
    Returns the log-likelihood using the estimate parameters
    """
    def loglikelihood(self):
        if not self._fitted_model:
            raise ValueError("You must fit the model before resampling")
        # Calculate likelihood
        loglikelihood = 0
        for i in range(self.num_samples):
            for j in range(self.num_params):
                if abs(self.X[i, j]) < .0006:
                    loglikelihood += np.log(1 - self.mean_theta_params[self.mean_class_assignments[i], j])
                else:
                    loglikelihood += np.log(self.mean_theta_params[self.mean_class_assignments[i], j])
        return loglikelihood

    """
    Initializes all the parameters.
    ----------------------
    num_samples: int (optional)
        the number of new samples in the data. Default will use the number of samples from fitted data
    num_params: int (optional)
        the number of parameter in the data. Default will use the number of samples from fitted data
    """
    def initialize_params(self,n_samples,num_params):
        theta_params = self._rand_num_gen.uniform(size=(self.num_classes,num_params))
        class_assignments = self._rand_num_gen.integers(0,self.num_classes,size=(n_samples))
        self.class_membership_scores = np.ones((n_samples,self.num_classes))
        class_probabilities = self._rand_num_gen.uniform(size=(self.num_classes))
        class_probabilities = class_probabilities / np.sum(class_probabilities)
        return theta_params,class_probabilities,class_assignments