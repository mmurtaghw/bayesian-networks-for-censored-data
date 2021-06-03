import pandas as pd
from pomegranate import *
import numpy as np
import matplotlib.pyplot as plt
import pygraphviz
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from feature_engine.discretisation import EqualWidthDiscretiser
import sklearn.metrics as metrics
from sklearn.metrics import classification_report


paramLearnWeights = False
structure_learning_with_weights = False
dependent_variable = "fustat"
factors_to_use = ["fustat", "surgery", "transplant","hla.a2","new_binned_mscore","new_binned_age", "reject", "outcome", "Tstar"]

data_file =pd.read_csv('bndata_zupan.csv')

for index, row in data_file.iterrows():
    data_file['birth.dt'][index] = int(row['birth.dt'][-2] + row['birth.dt'][-1])


for index, row in data_file.iterrows():
    data_file['age'][index] = int(row['age'])

data_file['mscore'] = data_file['mscore'].fillna(1)


disc = EqualWidthDiscretiser(bins=8, variables=['age','mscore'])
disc.fit(data_file)
new_binned_age_data = disc.transform(data_file)

# disc = EqualFrequencyDiscretiser(q=10, variables=['birth.dt'])
# disc.fit(data_file)
# new_binned_birthdt = disc.transform(data_file)




data_file['new_binned_age'] = 'e'
data_file['new_binned_age'] = new_binned_age_data['age']
data_file['new_binned_mscore'] = 'e'
data_file['new_binned_mscore'] = new_binned_age_data['mscore']


newdata = data_file[factors_to_use + ["KMW"]]
train, test, y_train, y_test = train_test_split(newdata, newdata, test_size=.80, random_state = 20)

weights = train[["KMW"]].to_numpy()
weights = np.reshape(weights, (np.product(weights.shape),))
train = train[factors_to_use]

position_of_dependent_variable = newdata.columns.get_loc(dependent_variable)
test = test[factors_to_use]
test_y = test[dependent_variable]
test[dependent_variable] = None



print("The shape of train is ",train.shape)
print("The shape of test is ",test.shape)

print("The shape of weights is", weights.shape)

if structure_learning_with_weights == False:
    model = BayesianNetwork.from_samples(train)
else:
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
predicted = np.array(predict).reshape(k[0],1)





acc = accuracy_score(predicted, test_y)
confusionMTRX = confusion_matrix(test_y,predicted)
f1_score =   classification_report(predicted, test_y)


if paramLearnWeights == True or structure_learning_with_weights == True:
    print("\n These results were acquired using zupan weights during the training of the bayesian network\n") 
    print("The accuracy of the model trained  using weights is \n",acc)
    print("\n The confusion matrix of the model trained  using weights is  \n",confusionMTRX)
    print("\n Other Classification metrics of the model trained  using weights is \n", f1_score)

else:
    print("\n No censoring weights were used while training the bayesian network\n") 
    print("The accuracy of the model trained without using weights is \n",acc)
    print("\n The confusion matrix of the model trained without using weights is  \n",confusionMTRX)
    print("\n Other Classification metrics of the model trained without using weights is \n", f1_score)




model.plot()
plt.savefig('zupan_bn.png')