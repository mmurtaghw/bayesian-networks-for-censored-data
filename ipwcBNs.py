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
paramLearnWeights = False
dependent_variable = "fustat"
factors_to_use = ["fustat",  "transplant", "reject", "surv"]

data_file =pd.read_csv('bndata_inverse.csv')



for index, row in data_file.iterrows():
    data_file['birth.dt'][index] = (row['birth.dt'][-2] + row['birth.dt'][-1])
    data_file['age'][index] = int(row['age'])

disc = EqualFrequencyDiscretiser(q=10, variables=['age'])
disc.fit(data_file)
new_binned_age_data = disc.transform(data_file)

data_file['new_binned_age'] = 'e'
data_file['new_binned_age'] = new_binned_age_data['age']


newdata = data_file[factors_to_use + ["weights"]]
train, test, y_train, y_test = train_test_split(newdata, newdata, test_size=.90, random_state = 20)

weights = train[["weights"]].to_numpy()
weights = np.reshape(weights, (np.product(weights.shape),))
train = train[factors_to_use]

position_of_dependent_variable = newdata.columns.get_loc(dependent_variable)
test = test[factors_to_use]
test_y = test[dependent_variable]
test[dependent_variable] = None



print("The shape of train is ",train.shape)
print("The shape of test is ",test.shape)

print("The shape of weights is", weights.shape)


model = BayesianNetwork.from_samples(train, weights=weights)
if paramLearnWeights == True:
    print("Parameter is learning is taking place but with weights this time")
    model.fit(train, weights=weights, n_jobs = 1)


test = test.to_numpy()
res = model.predict(test)

# print("The predicted values are for the test set ")
# for prediction in res:
#     print(prediction[0])


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



