from MainPackage import MixtureModelBernoulli
import pandas as pd
import time

num_classes = 4
random_state = 100

# Get data
df = pd.read_csv("../Data/test/test_data.csv", index_col=0)

# Set up model
C_list = MixtureModelBernoulli(num_classes=num_classes,
                               random_state=random_state,
                               burn_in=0,
                               max_iter=1000)


# fit the data to the model
start_time = time.time()
C_list.fit(df)
total_time = time.time() - start_time

print("DONE!")
print(f"TOTAL TIME{total_time}")

# Get parameters and other things
k, theta, pi = C_list.get_params()
print("pi")
print(pi)

print('\n----------\n')

print("theta")
print(theta)

print('\n----------\n')

print("k")
print(k)


print('\n----------\n')
print("Membership scores")
print(C_list.get_class_membership_scores())