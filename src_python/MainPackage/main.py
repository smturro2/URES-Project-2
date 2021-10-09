import numpy as np
import pandas as pd
from .main_helpers import *


class MixtureModelBernoulli():

    """
    Class Constructor
    """
    def __init__(self,num_classes=1,tol=.001,max_iter=1000,random_state=None,burn_in=100):
        self.num_classes = num_classes
        self.tol = tol
        self.max_iter = max_iter
        self.burn_in = burn_in
        # Boolean to see if model was fitted
        self.fitted_model = False
        # Set up random number generator. This is used to sample values
        self.rand_num_gen = np.random.default_rng(random_state)



    # Functions used to train the model

    """
    fit the model to the given data X
    X : array-like of shape (n_samples, n_features)
    """
    def fit(self,X):
        # Filter and check the input
        X = check_input_X(X)
        num_samples = X.shape[0]
        num_params = X.shape[1]

        # Initialize Parameters
        self.initialize_params(num_samples,num_params)

        # Initialize lists for samples
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
        toleranceMet = False
        while num_iter < (self.max_iter+self.burn_in) and not toleranceMet:

            # Calculate num_members_per_class
            num_members_per_class = np.bincount(self.class_assignments,minlength=self.num_classes)


            # Calculate num_correct_per_class_by_q
            # todo double check this
            num_correct_per_class_by_q = np.zeros((self.num_classes,num_params))
            for i in range(num_samples):
                num_correct_per_class_by_q[self.class_assignments[i]] += X[i]


            # Sample theta_params
            for c in range(self.num_classes):
                for j in range(num_params):
                    alpha = theta_hyperparams[0]+num_correct_per_class_by_q[c,j]
                    beta = theta_hyperparams[0]+num_members_per_class[c]-num_correct_per_class_by_q[c,j] # todo double check
                    self.theta_params[c,j] = self.rand_num_gen.beta(alpha,beta)
            # Append to samples list
            if num_iter>self.burn_in:
                self.samples_theta_params.append(self.theta_params.copy())


            # Calculate class_membership_scores
            self.class_membership_scores = np.ones_like(self.class_membership_scores)
            for i in range(num_samples):
                for c in range(self.num_classes):
                    self.class_membership_scores[i, c] = np.prod(self.theta_params[c,:]**X[i,:] * (1-self.theta_params[c,:])**(1-X[i,:])) * self.class_probabilities[c]
            # Normalize
            self.class_membership_scores = self.class_membership_scores / np.sum(self.class_membership_scores,axis=1,keepdims=True)
            # Append to samples list
            if num_iter>self.burn_in:
                self.samples_class_membership_scores.append(self.class_membership_scores.copy())


            # Sample class_assignments
            for i in range(num_samples):
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
    """
    def fit_predict(self,X):
        self.fit(X)
        return self.get_params()[0]


    # Main Methods

    """
    """
    def get_params(self):
        if not self.fitted_model:
            raise ValueError("You must fit the model before asking for the parameters")

        # Find means
        self.mean_theta_params = np.average(self.samples_theta_params,axis=0)
        self.mean_class_assignments = np.average(self.samples_class_assignments,axis=0)
        self.mean_class_probabilities = np.average(self.samples_class_probabilities,axis=0)

        return self.mean_class_assignments, self.mean_theta_params, self.mean_class_probabilities

    """
    """
    def get_class_membership_scores(self):
        if not self.fitted_model:
            raise ValueError("You must fit the model before asking for the parameters")

        # Find means
        self.mean_class_membership_scores = np.average(self.samples_class_membership_scores,axis=0)
        return self.class_membership_scores

    """
    Samples from the kth class
    """
    def sample(self, k):
        if not self.fitted_model:
            raise ValueError("You must fit the model before sampling from a latent class")
        # todo
        pass

    # Functions not typically used by user

    """
    Initializes the all the parameters.
    """
    def initialize_params(self,n_samples,num_params):
        self.theta_params = self.rand_num_gen.uniform(size=(self.num_classes,num_params))
        self.class_assignments = self.rand_num_gen.integers(0,self.num_classes,size=(n_samples))
        self.class_membership_scores = np.ones((n_samples,self.num_classes))
        self.class_probabilities = self.rand_num_gen.uniform(size=(self.num_classes))
        self.class_probabilities = self.class_probabilities / np.sum(self.class_probabilities)