import numpy as np
class LinearRegression(): 
    # The init function is used to set the learning rate and the number of iterations for  gradient descent in order for  the model to predict the outcome accurately.

    def __init__(self,learning_rate=0.05,iterations=1000):
        self.learning_rate=learning_rate
        self.iterations=iterations
        self.weights=None
        self.bias=None

    # The fit function is used for training the model using the training data 
    def fit(self,X,y):
        m,n=X.shape 
        

        # Initialising the weights of all the features to zero.
        self.weights=np.zeros(n)
        
        # Initialising the bias to zero.
        self.bias=0


        for _ in range(self.iterations):
            y_pred=np.dot(X,self.weights)+self.bias 
            dw=(1/m)*np.dot(X.T,(y_pred-y))
            db=(1/m)*np.sum(y_pred-y)
            self.weights=self.weights-self.learning_rate*dw
            self.bias=self.bias-self.learning_rate*db
            y_pred = np.dot(X, self.weights) + self.bias
            mse = self.mse(y, y_pred)
            if mse <= 0.1:
                break

    # The predict function is used to predict the target variable using the test data 
    def predict(self,X):
        
        return np.dot(X,self.weights)+self.bias

    