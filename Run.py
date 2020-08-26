from LogisticRegression import logistic_regression
from LogisticRegression import weighted_logistic_regression
from LinerRegression import Liner_regression
from LinerRegression import weighted_liner_regression
from utility import datasets 

def test_liner(x_train,x_test,y_train,y_test):
    sigVec=[0.1 , 0.3 , 0.8 , 2 , 10 ]
    accurecy=Liner_regression(x_train,x_test,y_train,y_test)
    print("Liner Regression\nSigma\tAccurecy")
    print("------------------------------------------")
    print("simple\t| "+str(accurecy))
    for i in (sigVec):
        accurecy=weighted_liner_regression(x_train,x_test,y_train,y_test,weight_Sigma=i)
        print(str(i)+'\t| '+str(accurecy))

        
def test_logistic(x_train,x_test,y_train,y_test):
    sigVec=[ 0.01 , 0.05 , 0.1 , 0.5 , 0.8 ,  1,5 ]
    accurecy=logistic_regression(x_train,x_test,y_train,y_test)
    print(" Logistic Regression\nSigma\tAccurecy")
    print("------------------------------------------")
    print("simple\t| " + str(accurecy))
    for i in (sigVec):
        accurecy=weighted_logistic_regression(x_train,x_test,y_train,y_test,weight_Sigma=i)
        print(str(i)+'\t| '+str(accurecy))

if __name__ == '__main__':
    #x_train,x_test,y_train,y_test = datasets().Load_linear_data()
    #Liner_regression(x_train,y_train,x_test,y_test)

    #weighted_logistic_regression(x_train,x_test,y_train,y_test,weight_Sigma=1)
    #logistic_regression(x_train,x_test,y_train,y_test)
    #print()
    x_train,x_test,y_train,y_test = datasets().Load_linear_data()
    test_liner(x_train,x_test,y_train,y_test)
    x_train,x_test,y_train,y_test = datasets().Load_Logistic_datas()
    test_logistic(x_train,x_test,y_train,y_test)