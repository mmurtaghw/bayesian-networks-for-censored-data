import pandas as pd
from pomegranate import *
import numpy as np
import matplotlib.pyplot as plt
import pygraphviz
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


data_file =pd.read_csv('bndata.csv')
print(data_file.head)

newdata = data_file[["trt","sex", "death"]]
train, test, y_train, y_test = train_test_split(newdata, newdata, test_size=.25)


weights = data_file[["weights"]].to_numpy()
weights = np.reshape(weights, (np.product(weights.shape),))

# death = train[["death"]].to_numpy()
# death = np.reshape(death, (np.product(death.shape),))
# print("The shape of death is  ", death.shape)
print("The training data set is")
print(train)

model = BayesianNetwork.from_samples(train )
# print(model.score(data_file[["trt","sex", "death"]].to_numpy(), data_file[["rltime"]].to_numpy()))
# model.fit(newdata,weights=weights, n_jobs = 1)
deathclass = test["death"]
test = test.to_numpy()

res = model.predict(test)

predict = []
k = np.shape(res)
for i in range(k[0]):
    predict.append(res[i][2])
predict = np.array(predict).reshape(k[0],1)


acc = accuracy_score(predict, deathclass)

confusionMTRX = confusion_matrix(deathclass,predict)
print(acc)
print(confusionMTRX)

model.plot()
plt.savefig('bn.png')

print(deathclass[:20])
print(res[:20])





