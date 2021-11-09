import pandas as pd
import time
import sys
import numpy as np

# Import MainPackage
sys.path.append('../src_python')
from MainPackage import MixtureModelBernoulli

# Get data
df = pd.read_csv("../Data/test/test_data.csv", index_col=0)

# Run model
num_classes = 4
random_state = 100
max_iter = 200
burn_in = 100
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


num_simulations = 2  # Recommended 100
mc_samples_theta = []
mc_samples_pi = []
mc_samples_k = []
mc_samples_true_k = []
mc_samples_assignments = []
mc_samples_times = []


def cluster_mapping(x):
    return mapping[x]

for num_sim in range(num_simulations):
    # Sample new data
    mc_data, mc_k = C_list.resample()

    # Set up model
    C_temp = MixtureModelBernoulli(num_classes=num_classes,
                                   random_state=random_state,
                                   burn_in=burn_in,
                                   max_iter=max_iter)

    # fit the data to the model
    print(f"\nRunning simulation {num_sim + 1}/{num_simulations}...", end="")
    start_time = time.time()
    C_temp.fit(mc_data)
    total_time = time.time() - start_time
    print("DONE!")
    print(f"Total time: {round(total_time, 5)} (secs)")
    theta_temp, pi_temp, k_temp = C_temp.get_params()

    # Find Freq table for reindexing
    freq_table = pd.DataFrame()
    freq_table["True Class"] = C_list.mean_class_assignments
    freq_table["Pred Class"] = k_temp
    freq_table["Ones"] = 1
    freq_table = freq_table.pivot_table(columns="Pred Class",
                                        index="True Class",
                                        values="Ones",
                                        aggfunc=sum)
    freq_table = freq_table.fillna(0)

    # Reindex
    mapping = np.argmax(freq_table.to_numpy(), axis=0)
    mapping_inv = np.argmax(freq_table.to_numpy(), axis=1)
    k_temp = mapping[k_temp]
    pi_temp = pi_temp[mapping_inv]
    theta_temp = theta_temp[mapping_inv]

    mc_samples_theta.append(theta_temp)
    mc_samples_pi.append(pi_temp)
    mc_samples_k.append(k_temp)
    mc_samples_true_k.append(mc_k)
    mc_samples_assignments.append(C_temp.get_class_membership_scores())
    mc_samples_times.append(total_time)
print("----------------------")
print("DONE!!!")
print(f"Ran {num_simulations} Simulations in {round(np.sum(mc_samples_times), 3)} total secs ")
print(f"Average: {np.sum(mc_samples_times) / num_simulations} (secs) ")