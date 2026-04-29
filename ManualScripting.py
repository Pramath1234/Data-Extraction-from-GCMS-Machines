import numpy as np
import json

# Load JSON file
with open("Greedy.json", "r") as f:
    data = json.load(f)

# Input file name to search
file_name = input("Enter the file name (e.g. 12038A.pdf): ")

# Find the matching entry
entry = next((item for item in data if item["file"] == file_name), None)

if entry is None:
    print(f"File '{file_name}' not found in JSON.")
else:
    print(f"Chemical: {entry['chemical_name']}")
    print(f"Spectrum:            {entry['spectrum']}")
    
    predicted = np.array(entry["relative_abundance"])
    print(f"Predicted Abundance: {predicted}")

    # Input array A and max height H
    a = list(map(float, input("Enter the values for array A: ").split()))
    H = float(input("Enter the max height H: "))

    # Calculate Actual relative abundance
    actual = np.array([(x / H) * 100 for x in a])
    print(f"Actual Abundance:    {actual}")

    # Validate lengths match
    if len(actual) != len(predicted):
        print(f"Length mismatch: Actual has {len(actual)} values, Predicted has {len(predicted)} values.")
    else:
        # Calculate and print RMSE
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        print(f"RMSE: {rmse:.4f}")