import pandas as pd

from MainPackage import MixtureModelBernoulli
import numpy as np

num_classes = 2
random_state = 100


# Testing List
X_list = [[1,0,1,0],
          [0,1,0,1],
          [1,0,1,0]]
C_list = MixtureModelBernoulli(num_classes=num_classes,
                               random_state=random_state,burn_in=0)
C_list.fit(X_list)


print(C_list.get_params()[0])
print('\n----------\n')
print(C_list.get_params()[1])
print('\n----------\n')
print(C_list.get_params()[1])

print(C_list.get_class_membership_scores())