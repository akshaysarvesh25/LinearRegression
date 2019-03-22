import numpy as np
from sklearn.linear_model import LinearRegression
import random
from operator import itemgetter



"""
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

#print(y_test)

x_train_ip = np.concatenate((x_train_1,x_train_2),axis=1)

x_test_ip  = np.concatenate((x_test_1,x_test_2),axis=1)

#random.sample(range(1, len(x_train_ip)), 3)

reg = LinearRegression().fit(x_train_ip, y_train)

y_pred = reg.predict(x_test_ip)



y_pred = [-1 if (x<=0) else 1 for x in y_pred]

y_pred = np.array(y_pred).reshape(-1,1)
y_test = np.array(y_test).reshape(-1,1)

#print((y_pred != y_test).sum())
print(x_train_ip)
"""



class DataGet:
    def __init__(self,sample_train,sample_test):
        self.x_train_path = sample_train
        self.x_test_path  = sample_test

    def GetSamples(self):
        x_train = np.genfromtxt(self.x_train_path,delimiter=',')
        x_train = np.array(x_train)
        x_test = np.genfromtxt(self.x_test_path,delimiter=',')
        x_test = np.array(x_test)

        return x_train,x_test

    def CollateData(self):
        x_train,x_test = self.GetSamples()
        y_train =  [sub_list[2] for sub_list in x_train]
        x_train_1 = [[sub_list[0]] for sub_list in x_train]
        x_train_2 = [[sub_list[1]] for sub_list in x_train]

        y_test = [sub_list[2] for sub_list in x_test]
        x_test_1 = [[sub_list[0]] for sub_list in x_test]
        x_test_2 = [[sub_list[1]] for sub_list in x_test]

        x_train_ip = np.concatenate((x_train_1,x_train_2),axis=1)
        x_test_ip  = np.concatenate((x_test_1,x_test_2),axis=1)

        y_test = np.array(y_test).reshape(-1,1)
        y_train = np.array(y_train).reshape(-1,1)

        return x_train_ip,y_train,x_test_ip,y_test



def RandomizedTrainingDataGenerator(x_train_ip,y_train_ip,NumbSamples):
    Samples = random.sample(range(1, len(x_train_ip)), NumbSamples)
    return np.array(itemgetter(*Samples)(x_train_ip)),np.array(itemgetter(*Samples)(y_train_ip))

def LinearRegressionImplement(x_train_ip,y_train_ip,NumbSamples):
    x,y = RandomizedTrainingDataGenerator(x_train_ip,y_train_ip,NumbSamples)
    return LinearRegression().fit(x,y)

def LinearRegressionFit(x_test_ip,y_test_ip,LinearRegObj):
    return LinearRegObj.predict(x_test_ip)

def LinearRegressionErrorCheck(y_test,y_pred):
    return ((y_pred != y_test).sum()/2000)

Train_data_numbers = [10, 50, 100, 500]

InputPath_trainingSet = '../Gamma_train.txt'
InputPath_testingSet = '../Gamma_test.txt'
DataObj = DataGet(InputPath_trainingSet,InputPath_testingSet)

x_train_ip,y_train_ip,x_test_ip,y_test_ip = DataObj.CollateData()

LinReg = [LinearRegressionImplement(x_train_ip,y_train_ip,sample_points) for sample_points in (Train_data_numbers)]
FittedOuput = [LinearRegressionFit(x_test_ip,y_test_ip,LinRegObjs) for LinRegObjs in (LinReg)]
print((FittedOuput))
err = [LinearRegressionErrorCheck(y_test_ip,FittedOuput) for LinRegObjs in (LinReg)]
print(err)




"""
def main():

    d = DataGet(a,b)
    d.GetSamples()

    e,f,g,h = d.CollateData()

if __name__ == "__main__":
    main()
"""
