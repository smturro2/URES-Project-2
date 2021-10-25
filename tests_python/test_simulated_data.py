import pandas as pd
import time
import sys

# Import MainPackage
sys.path.append('../src_python')
from MainPackage import MixtureModelBernoulli


num_classes = 4
random_state = 100

# Get data
df = pd.read_csv("../Data/test/test_data.csv", index_col=0)

# Set up model
max_iter = 500
burn_in = 0
C_list = MixtureModelBernoulli(num_classes=num_classes,
                               random_state=random_state,
                               burn_in=burn_in,
                               max_iter=max_iter)

# fit the data to the model
start_time = time.time()
C_list.fit(df)
total_time = time.time() - start_time

print("DONE!")
print(f"TOTAL TIME {total_time}")

# Get parameter estimates
theta, pi, k = C_list.get_params()
print(f"theta: {theta.shape}")
print(f"pi: {pi.shape}")
print(f"k: {k.shape}")