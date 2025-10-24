from linear_regression import LinearRegression  
import numpy as np 

np.random.seed(42)
n_samples = 1000
X = np.random.rand(n_samples, 5)*10
noise = np.random.randn(n_samples) * 0.5
true_weights = np.array([5 , 1 , 2 , 2 , 3])
true_bias = 2
y = X@true_weights + true_bias + noise 

model = LinearRegression(learning_rate = 0.002 , n_iterations = 40000)
print("First 5 values of data:")
print(X[:5])
print("\nFirst 5 values of y:")
print(y[:5])
model.fit(X , y)

