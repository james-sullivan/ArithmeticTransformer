import os

import pandas as pd
import random
import numpy as np
from sklearn.model_selection import train_test_split

def generate_addition_problem(modulus, max_number, rng):
    a = rng.integers(0, max_number + 1)
    b = rng.integers(0, max_number + 1)
    c = rng.integers(0, max_number + 1)

    coinFlip = 2 # random.randint(1,2)
    if coinFlip == 1:
        result = (a * b) % modulus
        input = f"( {a} * {b} ) % {modulus}"
        operation = "multiplication"
    else:
        result = (a + b) % modulus
        input = f"{a} {b} {modulus}"
        operation = "addition"

    '''randInt = rng.integers(1, 5)
    if randInt == 1:
        result = (a * b * c) % modulus
        input = f"({a} * {b} * {c}) % {modulus}"
        operation = "multiplication"
    elif randInt == 2:
        result = (a + b + c) % modulus
        input = f"({a} + {b} + {c}) % {modulus}"
        operation = "addition"
    elif randInt == 3:
        result = (a * b + c) % modulus
        input = f"({a} * {b} + {c}) % {modulus}"
        operation = "addAndMulti"
    else:
        result = (a + b * c) % modulus
        input = f"({a} + {b} * {c}) % {modulus}"
        operation = "addAndMulti"'''

    return {
        "input": input,
        "output": str(result),
        "modulus": modulus,
        "operation": operation
    }

def generate_unique_dataset(num_problems, moduli_range, max_number, rng):
    unique_problems = set()
    data = []
    while len(data) < num_problems:
        modulus = random.randint(moduli_range[0], moduli_range[1])
        problem = generate_addition_problem(modulus, max_number, rng)
        if problem['input'] not in unique_problems:
            unique_problems.add(problem['input'])
            data.append(problem)
    return pd.DataFrame(data)

# Configuration
total_problems = 12000
test_size = 0.2
moduli_range = (10, 10)
max_number = 99
random_state = 42  # For reproducibility
rng = np.random.default_rng(random_state)
data_set_name = "modular_arithmetic_three_numbers"

'''
available_unique_problems = (moduli_range[1] - moduli_range[0] + 1) * max_number * max_number

# Make sure there are enough problems
if 0.95 * available_unique_problems < total_problems:
    print(f"WARNING: With the given parameters, there might not be enough problems to create. available_unique_problems = {available_unique_problems}")
'''

# Generate a single unique dataset
print(f"Generating {total_problems} unique problems...")
df = generate_unique_dataset(total_problems, moduli_range, max_number, rng)
print(f"Generated {len(df)} unique problems.")

# Split the dataset into training and test sets
train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)

# Create the data folder if it doesn't exist
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "..", "data")
data_path = os.path.abspath(data_path)
os.makedirs(data_path, exist_ok=True)

# Save to CSV
train_name = f"{data_set_name}_train.csv"
test_name = f"{data_set_name}_test.csv"
train_path = os.path.join(data_path, train_name)
test_path = os.path.join(data_path, test_name)
train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

print(f"Training data ({len(train_df)} problems) saved to {train_name}")
print(f"Test data ({len(test_df)} problems) saved to f{test_name}")

# Verify no overlap (this should always be true, but it's good to check)
train_inputs = set(train_df['input'])
test_inputs = set(test_df['input'])
overlap = train_inputs.intersection(test_inputs)
print(f"\nNumber of overlapping problems: {len(overlap)}")
assert len(overlap) == 0, "Error: There should be no overlap between training and test sets."
