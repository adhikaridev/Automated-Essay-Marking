import numpy as np
import pandas as pd
import sys

f = open("filename.txt")
f.readline()  # skip the header
data = np.loadtxt(f)
print(data)
