import pandas as pd
import numpy as np

num_classes = 2
random_state = 100


# Testing List
X_list = [[1,0,1,0],[0,1,0,1],[1,1,1,0]]
C_list = MixtureModelBernoulli(num_classes=num_classes,random_state=random_state)
C_list.fit(X_list)


# Testing Numpy
X_np = np.array(X_list)
C_np = MixtureModelBernoulli(num_classes=num_classes,random_state=random_state)
C_np.fit(X_np)

# Testing dataframe
X_df = pd.DataFrame(X_np,columns=["Question 1","Question 2","Question 3","Question 4"])
C_df = MixtureModelBernoulli(num_classes=num_classes,random_state=random_state)
C_df.fit(X_df)

print(C_list.get_params()[2])
print(C_np.get_params()[2])
print(C_df.get_params()[2])

print('\n----------\n')

print(C_list.get_class_membership_scores())
print(C_np.get_class_membership_scores())
print(C_df.get_class_membership_scores())