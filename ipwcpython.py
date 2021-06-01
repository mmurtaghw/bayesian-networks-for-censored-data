import pandas as pd
from pomegranate import *
import numpy as np
import pygraphviz 

data_file =pd.read_csv('bndata.csv')
print(data_file.head)

newdata = data_file[["trt","sex", "death"]].to_numpy()
# print(data_file[["weights"]])
# print(newdata.head)

weights = data_file[["weights"]].to_numpy()
print(weights.flatten())
print(weights.shape)

weights = np.reshape(weights, (np.product(weights.shape),))
print(weights.shape)

death = data_file[["death"]].to_numpy()
death = np.reshape(death, (np.product(death.shape),))
print("The shape of death is  ", death.shape)


model = BayesianNetwork.from_samples(newdata, algorithm='exact',weights= weights )
# print(model.score(data_file[["trt","sex"]].to_numpy(), death))

print(model.predict([[1, 2, None]]))

model.plot()