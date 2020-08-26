import numpy as np
import matplotlib.pyplot as plt
from utility import datasets 
from utility import Calculate_logistic_weights
from utility import sigmoid

#x_train,x_test,y_train,y_test = datasets().Load_Logistic_datas()
#x1=np.array([[0,0],[2,3],[2,1]])
#q=np.array([[2,0]])

# Calculate_logistic_weights(x1,q,0.8)
def indexed_weighted_logistic_regression(x_train,x_test,y_train,y_test,num_iter=10000,Regulization_lamda=0.0001,lr=0.001,weight_Sigma=0.8):
    weights=Calculate_logistic_weights(x_train,x_test,weight_Sigma)
    weights =np.identity(weights.shape[0]) * weights
    #weights=np.reshape(weights,(weights.shape[0],1))
    
    
    theta = np.reshape(np.zeros(x_train.shape[1]),(x_train.shape[1],1))
    for i in range(num_iter):
        z = np.dot(x_train, theta)
        h = sigmoid(z)
        h = np.reshape(h,(h.shape[0],1))
        #print(x_train.shape)
        #print(weights.shape)
        X=np.matmul(weights,x_train)
        gradient = -np.dot(X.T, (h - y_train)) + (theta) * Regulization_lamda
        #gradient = -np.dot(np.matmul(weights,x_train).T, (h - y_train)) + (theta) * Regulization_lamda
        theta = theta + lr * gradient
    y_pred=sigmoid(np.dot(x_test.T, theta))
    #print(abs(y_test - y_pred),end='#############################\n')
    return y_pred

def weighted_logistic_regression(x_train,x_test,y_train,y_test,num_iter=10000,Regulization_lamda=0.0001,lr=0.001,weight_Sigma=0.8):
    acc=0
    for i in range(x_test.shape[0]):
        #print(type(x_test[i]))
        y_predicted_i = indexed_weighted_logistic_regression(x_train,np.reshape(x_test[i,:],(2,1)),y_train,y_test[i],weight_Sigma=weight_Sigma,num_iter=10000,Regulization_lamda=0.0001,lr=0.001)
        if((y_predicted_i>0.5 and y_test[i]==1)or(y_predicted_i<0.5 and y_test[i]==0)):
            acc+=1
        
    acc = acc/y_test.shape[0]
    #print("weighted Accurecy : "+str(acc))


    return 100 * acc

def logistic_regression(x_train,x_test,y_train,y_test,num_iter=10000,Regulization_lamda=0.0001,lr=0.001):
    theta = np.reshape(np.zeros(x_train.shape[1]),(x_train.shape[1],1))

    for i in range(num_iter):
        z = np.dot(x_train, theta)
        h = sigmoid(z)
        h = np.reshape(h,(h.shape[0],1))
        gradient = -np.dot(x_train.T, (h - y_train)) + (theta) * Regulization_lamda
        #gradient = -np.dot(X.T, (h - y))     
        theta = theta + lr * gradient

    z = np.dot(x_test, theta)
    h = sigmoid(z)
    h = np.reshape(h,(h.shape[0],1))
    acc=0
    for i in range (len(y_test)):
      if((h[i]>0.5 and y_test[i]==1)or(h[i]<0.5 and y_test[i]==0)):
          acc+=1
    acc = acc/y_test.shape[0]
    #print("simple Accurecy : "+str(acc))
    return   100*acc 


if __name__ == '__main__':
    x_train,x_test,y_train,y_test = datasets().Load_Logistic_datas()
    plt.scatter(x_train[:,0], x_train[:,1], c=y_train[:,0], cmap=plt.cm.Spectral)
    #plt.show()
    weighted_logistic_regression(x_train,x_test,y_train,y_test)
    logistic_regression(x_train,x_test,y_train,y_test)
    
