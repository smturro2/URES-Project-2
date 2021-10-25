import pandas as pd
import time
import sys

# Import MainPackage
sys.path.append('../src_python')
from MainPackage import MixtureModelBernoulli


df = pd.read_csv("../Data/NSI/narcissistic_personality_inventory.csv",index_col=0)

num_classes = 4
random_state = 100
max_iter = 300
burn_in = 100

print("Data Size")
print("---------")
print(f"Samples: {df.shape[0]}")
print(f"Features: {df.shape[1]}")

# Set up model
C_list = MixtureModelBernoulli(num_classes=num_classes,
                               random_state=random_state,
                               burn_in=burn_in,
                               max_iter=max_iter)


# fit the data to the model
print(f"\nRunning {max_iter} iterations...",end="")
start_time = time.time()
C_list.fit(df)
total_time = time.time() - start_time

print("DONE!")
print(f"Total time: {round(total_time,5)} (secs)")

# Get parameter estimates
theta, pi, k = C_list.get_params()
print(f"theta: {theta.shape}")
print(f"pi: {pi.shape}")
print(f"k: {k.shape}")


# Monte Carlo simulations
num_simulations = 2
mc_samples_theta = []
mc_samples_pi = []
mc_samples_k = []
mc_samples_true_k = []
mc_samples_assignments = []
mc_samples_times = []

for num_sim in range(num_simulations):
    # Sample new data
    mc_data, mc_k = C_list.resample(500)

    # Set up model
    C_temp = MixtureModelBernoulli(num_classes=num_classes,
                                   random_state=random_state,
                                   burn_in=burn_in,
                                   max_iter=max_iter)

    # fit the data to the model
    print(f"\nRunning simulation {num_sim}/{num_simulations}...", end="")
    start_time = time.time()
    C_temp.fit(mc_data)
    total_time = time.time() - start_time
    print("DONE!")
    print(f"Total time: {round(total_time, 5)} (secs)")

    # Append simulation data
    theta_temp, pi_temp, k_temp = C_temp.get_params()
    mc_samples_theta.append(theta_temp)
    mc_samples_pi.append(pi_temp)
    mc_samples_k.append(k_temp)
    mc_samples_true_k.append(mc_k)
    mc_samples_assignments.append(C_temp.get_class_membership_scores())
    mc_samples_times.append(total_time)