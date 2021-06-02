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


invert_predictions = False
paramLearnWeights = True
predict = []

data_file =pd.read_csv('bndata_zupan.csv')


for index, row in data_file.iterrows():
    data_file['birth.dt'][index] = (row['birth.dt'][-2] + row['birth.dt'][-1])
    data_file['age'][index] = int(row['age'])

disc = EqualFrequencyDiscretiser(q=10, variables=['age'])
disc.fit(data_file)

new_binned_age_data = disc.transform(data_file)

data_file['new_binned_age'] = 'initialization Variable'
data_file['new_binned_age'] = new_binned_age_data['age']
print(new_binned_age_data.head(20))


print(data_file[["new_binned_age", "age"]].head(30))



newdata = data_file[["fustat", "new_binned_age","surgery", "transplant", "hla.a2","reject","Tstar","outcome"]]
train, test, y_train, y_test = train_test_split(newdata, newdata, test_size=0.5, random_state = 20)


weights = data_file[["KMW"]].to_numpy()
weights = np.reshape(weights, (np.product(weights.shape),))

model = BayesianNetwork.from_samples(train)
if paramLearnWeights == True:
    model.fit(newdata,weights=weights, n_jobs = 1)

dependent_variable = "fustat"

test_y = test[dependent_variable]
position_of_dependent_variable = newdata.columns.get_loc(dependent_variable)

test = test.to_numpy()

res = model.predict(test)

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


print("The position of the depend variable is ", position_of_dependent_variable)

model.plot()
plt.savefig('zupan_bn.png')






# print(model.score(data_file[["trt","sex", "death"]].to_numpy(), data_file[["rltime"]].to_numpy()))
