import matplotlib.pyplot as plt
import numpy as np
from utility import datasets 
from utility import Calculate_liner_weights




def indexed_weighted_liner_regression(x_train,x_test,y_train,y_test,weight_Sigma=0.8):
    weights = Calculate_liner_weights(x_train,x_test,weight_Sigma)
    weights =np.identity(weights.shape[0])*weights
    
    #print(weights)
    x_train=np.concatenate((np.ones((len(x_train),1)),x_train),axis=1)
    #print(x_train)
    theta= np.matmul(np.matmul(np.matmul(np.linalg.inv(np.matmul(np.matmul(x_train.T,weights),x_train)),x_train.T),weights),y_train)
    pred=x_test * theta[1]+theta[0]
    return abs(y_test[0] - pred[0])

def weighted_liner_regression(x_train,x_test,y_train,y_test,weight_Sigma=0.8):
    Loss_Sum=0
    for i in range(x_test.shape[0]):
        Loss_Sum+=indexed_weighted_liner_regression(x_train,x_test[i],y_train,y_test[i],weight_Sigma)
   
    return 100 -(100*(Loss_Sum/x_test.shape[0]))



def Liner_regression(x_train,x_test,y_train,y_test):
    #phi_train=np.concatenate((np.ones((len(x_train),1)),x_train,np.power(x_train,3),np.power(x_train,5),np.power(x_train,6),np.power(x_train,7),np.power(x_train,8),np.power(x_train,9)),axis=1)
    phi_train=np.concatenate((np.ones((len(x_train),1)),x_train,np.power(x_train,3)),axis=1)

    # Training Step
    weight=np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(phi_train),phi_train)),np.transpose(phi_train)),y_train)


    # Test Phase
    #phi_test=np.concatenate((np.ones((len(x_test),1)),x_test,np.power(x_test,3),np.power(x_test,5),np.power(x_test,6),np.power(x_test,7),np.power(x_test,8),np.power(x_test,9)),axis=1)
    phi_test=np.concatenate((np.ones((len(x_test),1)),x_test,np.power(x_test,3)),axis=1)
    y_pred = np.matmul(phi_test,weight)

    
    # Explained variance score: 1 is perfect prediction
    #print(len(x_train))
    #print(len(x_test))
    # Plot outputs 

    #plt.scatter(x_train, y_train,  color='black')
    #plt.scatter(x_test, y_pred, color='blue')
    #plt.plot(x_test, y_pred, color='blue',linewidth=3)
    #plt.plot(x_test, y_test, color='green')
    #plt.show()
    return 100 - (100 * np.sum(abs(y_test[:] - y_pred[:]))/x_test.shape[0])


if __name__ == '__main__':
    x_train,x_test,y_train,y_test = datasets().Load_linear_data()
    Liner_regression(x_train,y_train,x_test,y_test)
    #loss_sum = weighted_liner_regression(x_train,y_train,x_test,y_test,0.1) 
    #print(loss_sum)
    


        
