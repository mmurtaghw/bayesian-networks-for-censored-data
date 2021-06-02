import pandas as pd
from pomegranate import *
import numpy as np
import matplotlib.pyplot as plt
import pygraphviz
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from feature_engine.discretisation import EqualFrequencyDiscretiser



invert_predictions = True
paramLearnWeights = True
data_file =pd.read_csv('bndata_inverse.csv')


for index, row in data_file.iterrows():
    data_file['birth.dt'][index] = (row['birth.dt'][-2] + row['birth.dt'][-1])
    data_file['age'][index] = int(row['age'])

birthdate_data = data_file['birth.dt'].to_numpy()
birthdate_data = birthdate_data.reshape((len(birthdate_data),1))

disc = EqualFrequencyDiscretiser(q=10, variables=['age'])
disc.fit(data_file)

new_binned_age_data = disc.transform(data_file)

data_file['new_binned_age'] = 'e'
data_file['new_binned_age'] = new_binned_age_data['age']
print(new_binned_age_data.head(20))


print(data_file[["new_binned_age", "age"]].head(30))



newdata = data_file[["fustat", "new_binned_age","surgery", "transplant", "mismatch", "hla.a2"]]
train, test, y_train, y_test = train_test_split(newdata, newdata, test_size=.90, random_state = 20)


weights = data_file[["weights"]].to_numpy()
weights = np.reshape(weights, (np.product(weights.shape),))

print("The training data set is")
print(train)

model = BayesianNetwork.from_samples(train)
if paramLearnWeights == True:
    model.fit(newdata,weights=weights, n_jobs = 1)

test_y = test["surgery"]
test = test.to_numpy()

res = model.predict(test)

position_of_dependent_variable = 0

predict = []
k = np.shape(res)

for i in range(k[0]):
    predict.append(res[i][position_of_dependent_variable])
predict = np.array(predict).reshape(k[0],1)

if invert_predictions == True:
    for index, prediction in enumerate(predict): 
        if prediction == 0:
            predict[index] = 1
        else:
            predict[index] =0
acc = accuracy_score(predict, test_y)

confusionMTRX = confusion_matrix(test_y,predict)
print(acc)
print(confusionMTRX)

model.plot()
plt.savefig('inverse_bn.png')



# print(model.score(data_file[["trt","sex", "death"]].to_numpy(), data_file[["rltime"]].to_numpy()))



