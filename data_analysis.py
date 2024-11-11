#Preliminary data analysis for the dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pdb

#Load the first split of the dataset, stored in P12data\P12Data_1\split_1\test_physionet2012_1.npy
#my file is a npy file

train_data = np.load('P12data/P12Data_1/split_1/train_physionet2012_1.npy', allow_pickle=True)
test_data = np.load('P12data/P12Data_1/split_1/test_physionet2012_1.npy', allow_pickle=True)
val_data = np.load('P12data/P12Data_1/split_1/validation_physionet2012_1.npy', allow_pickle=True)
#Convert the data to a pandas dataframe

pdb.set_trace()

#Print the length of the data 

print(f"Length of train data: {len(train_data)}")
print(f"Length of test data: {len(test_data)}")
print(f"Length of validation data: {len(val_data)}")

print("Length of whole dataset: ", len(train_data) + len(test_data) + len(val_data))
