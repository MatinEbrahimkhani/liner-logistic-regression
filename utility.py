import numpy as np
class datasets:
    """ 
    This the object that can fetch the required Datasets 

    $$$
    to load the dataset for liner regression models call     "Load_linear_data() "  
    input  ->   no input required
    all of the output matrixes are numpy array instances
    output ->   x_train, x_test, y_train, y_test
    shapes ->   (80,1),  (20,1), (80,1) , (20,1)
    $$$
    .
    .
    . 
    $$$
    to load the dataset for logistic regression models call     "Load_logistic_data() "  
    input  ->   no input required
    all of the output matrixes are numpy array instances
    output ->   x_train, x_test, y_train, y_test
    shapes ->   (55,2),  (14,2), (55,1) , (14,2)

    """
    def __init__(self):
        pass
    def Load_linear_data(self):
        X=[]
        Features = open("features.txt", "r")
        for x in Features:
            X.append(eval(x))
        Y=[]
        Labels = open("labels.txt","r")
        for y in Labels:
            Y.append(eval(y))
        boundry=80
        x_train=np.array(X[:boundry])
        x_test=np.array(X[boundry:])
        y_train=np.array(Y[:boundry])
        y_test=np.array(Y[boundry:])
        return np.reshape(x_train,(x_train.shape[0],1)),np.reshape(x_test,(x_test.shape[0],1)),np.reshape(y_train,(y_train.shape[0],1)), np.reshape(y_test,(y_test.shape[0],1))
    def Load_Logistic_datas(self):
        X1=[]
        X2=[]

        Features = open("features_r.txt", "r")
   
        for x in Features:
            #print(x)
            #print(type(x))
            file_input=(x.split("  "))                       
            X1.append(eval(file_input[1]))
            X2.append(eval(file_input[2]))
        #X1=np.array(X1)
        #X2=np.array(X2)

        Y=[]
        Labels = open("labels_r.txt","r")
        for y in Labels:
            
            Y.append(eval(y))
        X=np.column_stack((X1, X2))
        boundry=55
        x_train = np.array(X[:boundry])
        x_test = np.array(X[boundry:])
        y_train=np.array(Y[:boundry])
        y_test=np.array(Y[boundry:])
        return x_train,x_test,                                                                                      np.reshape(y_train,(y_train.shape[0],1)),                                                                np.reshape(y_test,(y_test.shape[0],1))




    
def Calculate_liner_weights(x_train,query,sigma=0.8):
    """
    this function returns the weights for a testing data named query

    """
    result = np.exp(-pow((x_train[:] - query),2)/(2*sigma*sigma) )
    return result


  
def Calculate_logistic_weights(x_train,query,sigma=0.8):
    """
    this function returns the weights for a testing data named query

    """
    x0_Diff = x_train[:,0] - query[0,0]
    #print(x_train[:,1],end=" ")
    #print(query[1,0])
    x1_Diff = x_train[:,1] - query[1,0]
    result = np.exp(-(pow(x0_Diff,2)+pow(x1_Diff,2))/(2*sigma*sigma))
    return result


def sigmoid(z):
    return 1 / (1 + np.exp(-z))