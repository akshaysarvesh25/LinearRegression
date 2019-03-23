import numpy as np
from sklearn.linear_model import LinearRegression
import random
from operator import itemgetter


def RandomizedTrainingDataGenerator(x_train_ip,y_train_ip,NumbSamples):
    Samples_pos1 = np.array(random.sample(range(0, len(x_train_ip)/2), NumbSamples))
    Samples_neg1 = np.array(random.sample(range(len(x_train_ip)/2,len(x_train_ip)), NumbSamples))
    Samples_pos1 = np.hstack((Samples_pos1))#,Samples_neg1))
    Samples_neg1 = np.hstack((Samples_neg1))
    Samples = np.hstack((Samples_pos1,Samples_neg1))
    #print(Samples)
    return np.array(itemgetter(*Samples)(x_train_ip)),np.array(itemgetter(*Samples)(y_train_ip))

def PredictOutput(x_train_rand1,y_train_rand1):
    return LinearRegression().fit(x_train_rand1, y_train_rand1)

x_train = np.genfromtxt('../Gamma_train.txt',delimiter=',')
x_train = np.array(x_train)

x_test = np.genfromtxt('../Gamma_test.txt',delimiter=',')
x_test = np.array(x_test)

y_train =  [sub_list[2] for sub_list in x_train]
x_train_1 = [[sub_list[0]] for sub_list in x_train]
x_train_2 = [[sub_list[1]] for sub_list in x_train]

y_test = [sub_list[2] for sub_list in x_test]
x_test_1 = [[sub_list[0]] for sub_list in x_test]
x_test_2 = [[sub_list[1]] for sub_list in x_test]

x_train_ip = np.concatenate((x_train_1,x_train_1),axis=1)

x_test_ip  = np.concatenate((x_test_1,x_test_2),axis=1)

random.sample(range(1, len(x_train_ip)), 3)

reg = LinearRegression().fit(x_train_ip, y_train)

y_pred = reg.predict(x_test_ip)

y_pred = [-1 if (x<=0) else 1 for x in y_pred]

y_pred = np.array(y_pred).reshape(-1,1)
y_test = np.array(y_test).reshape(-1,1)

#print(((y_pred != y_test).sum()))



Train_data_numbers = [10, 50, 100, 500]
x_train_rand,y_train_rand = RandomizedTrainingDataGenerator(x_train_ip,y_train,Train_data_numbers[0])
print(x_train_rand)
print(y_train_rand)
reg = PredictOutput(x_train_rand,y_train_rand)
print(reg.predict(x_test_ip))
