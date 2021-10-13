from MainPackage import MixtureModelBernoulli

num_classes = 2
random_state = 100

# Get data
X_list = [[1,0,1,0],
          [0,1,0,1],
          [1,0,1,0]]

# Set up model
C_list = MixtureModelBernoulli(num_classes=num_classes,
                               random_state=random_state,
                               burn_in=0,
                               max_iter=1000)

# fit the data to the model
C_list.fit(X_list)

# Get parameters and other things
print(C_list.get_params()[0])
print('\n----------\n')
print(C_list.get_class_membership_scores())