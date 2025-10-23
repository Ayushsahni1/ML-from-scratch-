    import numpy as np 

    class LinearRegression:
        def __init__(self , learning_rate = 0.01 , n_iterations = 1000):
            self.bias = None 
            self.weights = None
            self.learning_rate = learning_rate
            self.n_iterations = n_iterations

        def predict(self , data_matrix):
            if self.bias != None and self.weights != None:
                return data_matrix @ self.weights + self.bias

        def error(self , data_matrix , prediction_target):
            y_hat = self.predict(data_matrix)
            return prediction_target - y_hat 

        def fit(self , data_matrix , prediction_target):
            sample_count ,feature_count = np.shape(data_matrix)
            self.bias = 0
            self.weights = np.zeros(feature_count)
            current_iteration = 0
            while (current_iteration < self.n_iterations):
                error = self.error(data_matrix , prediction_target)
                cost = error.T@error / sample_count
                print("Cost: " , cost)
                weight_gradient = 2/sample_count*data_matrix.T@error 
                bias_gradient = 2/sample_count * np.sum(error)
                self.weights = self.weights + self.learning_rate*weight_gradient
                self.bias = self.bias + self.learning_rate*bias_gradient 
                current_iteration += 1
        









