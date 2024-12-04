import pandas as pd
import numpy as np

# Load and clean the dataset
dataset_2 = pd.read_csv('NYC_Bicycle_Counts_2016.csv')
bridges = ['Brooklyn Bridge', 'Manhattan Bridge', 'Queensboro Bridge', 'Williamsburg Bridge']
for bridge in bridges:
    dataset_2[bridge] = pd.to_numeric(dataset_2[bridge].replace(',', '', regex=True))

# Create individual numpy arrays for each bridge's data
bridge_data = {bridge: dataset_2[bridge].to_numpy() for bridge in bridges}

# Find the standard error of each bridge's dataset
def sample_stde(data):
    return np.std(data, ddof=1) / np.sqrt(len(data))

bridge_stde = {bridge: sample_stde(data) for bridge, data in bridge_data.items()}

# Find the bridge with the highest STDE
max_stde_bridge = max(bridge_stde, key=bridge_stde.get)
print(f'Put sensors on every bridge but the {max_stde_bridge}')