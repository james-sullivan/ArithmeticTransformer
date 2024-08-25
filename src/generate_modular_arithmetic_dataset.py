import os

import pandas as pd
import random
from sklearn.model_selection import train_test_split

def generate_addition_problem(modulus, max_number):
    a = random.randint(0, max_number)
    b = random.randint(0, max_number)
    result = (a + b) % modulus
    return {
        "input": f"( {a} + {b} ) % {modulus} =",
        "output": str(result),
        "modulus": modulus,
        "operation": "addition"
    }

def generate_unique_dataset(num_problems, moduli_range, max_number):
    unique_problems = set()
    data = []
    while len(data) < num_problems:
        modulus = random.randint(moduli_range[0], moduli_range[1])
        problem = generate_addition_problem(modulus, max_number)
        if problem['input'] not in unique_problems:
            unique_problems.add(problem['input'])
            data.append(problem)
    return pd.DataFrame(data)

# Configuration
total_problems = 10000
test_size = 0.2
moduli_range = (10, 10)
max_number = 99
random_state = 42  # For reproducibility

available_unique_problems = (moduli_range[1] - moduli_range[0] + 1) * max_number * max_number

# Make sure there are enough problems
if 0.95 * available_unique_problems < total_problems:
    print(f"WARNING: With the given parameters, there might not be enough problems to create. available_unique_problems = {available_unique_problems}")

# Generate a single unique dataset
print(f"Generating {total_problems} unique problems...")
df = generate_unique_dataset(total_problems, moduli_range, max_number)
print(f"Generated {len(df)} unique problems.")

# Split the dataset into training and test sets
train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)

# Create the data folder if it doesn't exist
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "..", "data")
data_path = os.path.abspath(data_path)
os.makedirs(data_path, exist_ok=True)

# Save to CSV
train_path = os.path.join(data_path, "modular_arithmetic_train.csv")
test_path = os.path.join(data_path, "modular_arithmetic_test.csv")
train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

print(f"Training data ({len(train_df)} problems) saved to modular_arithmetic_train.csv")
print(f"Test data ({len(test_df)} problems) saved to modular_arithmetic_test.csv")

# Display some statistics
print("\nTraining Data Statistics:")
print(train_df['modulus'].describe())

print("\nTest Data Statistics:")
print(test_df['modulus'].describe())

# Verify no overlap (this should always be true, but it's good to check)
train_inputs = set(train_df['input'])
test_inputs = set(test_df['input'])
overlap = train_inputs.intersection(test_inputs)
print(f"\nNumber of overlapping problems: {len(overlap)}")
assert len(overlap) == 0, "Error: There should be no overlap between training and test sets."
