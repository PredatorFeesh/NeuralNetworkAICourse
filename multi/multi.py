# %%
#Importing the required libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#Sigmoid is used as the activation function
def sigmoid(x):
    x = np.array(x, dtype=np.float128())
    return 1/(1 + np.exp(-x))

#Derivative of the sigmoid function
def sigmoid_prime(x):
    return sigmoid(x)*(1.0 - sigmoid(x))

class NeuralNetwork(object):
    
    def __init__(self, architecture):
        #architecture - numpy array with ith element representing the number of neurons in the ith layer.
        
        #Initialize the network architecture
        self.L = architecture.size - 1 #L corresponds to the last layer of the network.
        self.n = architecture #n stores the number of neurons in each layer
        #input_size is the number of neurons in the first layer i.e. n[0]
        #output_size is the number of neurons in the last layer i.e. n[L]
        
        #Parameters will store the network parameters, i.e. the weights and biases
        self.parameters = {}
        
        #Initialize the network weights and biases:
        for i in range (1, self.L + 1): 
            #Initialize weights to small random values
            self.parameters['W' + str(i)] = np.random.randn(self.n[i], self.n[i - 1]) * 0.01
            
            #Initialize rest of the parameters to 1
            self.parameters['b' + str(i)] = np.ones((self.n[i], 1))
            self.parameters['z' + str(i)] = np.ones((self.n[i], 1))
            self.parameters['a' + str(i)] = np.ones((self.n[i], 1))
        
        #As we started the loop from 1, we haven't initialized a[0]:
        self.parameters['a0'] = np.ones((self.n[i], 1))
        
        #Initialize the cost:
        self.parameters['C'] = 1
        
        #Create a dictionary for storing the derivatives:
        self.derivatives = {}
                    
    def forward_propagate(self, X):
        #Note that X here, is just one training example
        self.parameters['a0'] = X
        
        #Calculate the activations for every layer l
        for l in range(1, self.L + 1):
            self.parameters['z' + str(l)] = np.add(np.dot(self.parameters['W' + str(l)], self.parameters['a' + str(l - 1)]), self.parameters['b' + str(l)])
            self.parameters['a' + str(l)] = sigmoid(self.parameters['z' + str(l)])
        
    def compute_cost(self, y):
        self.parameters['C'] = -(y*np.log(self.parameters['a' + str(self.L)]) + (1-y)*np.log( 1 - self.parameters['a' + str(self.L)]))
    
    def compute_derivatives(self, y):
        #Partial derivatives of the cost function with respect to z[L], W[L] and b[L]:        
        #dzL
        self.derivatives['dz' + str(self.L)] = self.parameters['a' + str(self.L)] - y
        #dWL
        self.derivatives['dW' + str(self.L)] = np.dot(self.derivatives['dz' + str(self.L)], np.transpose(self.parameters['a' + str(self.L - 1)]))
        #dbL
        self.derivatives['db' + str(self.L)] = self.derivatives['dz' + str(self.L)]

        #Partial derivatives of the cost function with respect to z[l], W[l] and b[l]
        for l in range(self.L-1, 0, -1):
            self.derivatives['dz' + str(l)] = np.dot(np.transpose(self.parameters['W' + str(l + 1)]), self.derivatives['dz' + str(l + 1)])*sigmoid_prime(self.parameters['z' + str(l)])
            self.derivatives['dW' + str(l)] = np.dot(self.derivatives['dz' + str(l)], np.transpose(self.parameters['a' + str(l - 1)]))
            self.derivatives['db' + str(l)] = self.derivatives['dz' + str(l)]
            
    def update_parameters(self, alpha):
        for l in range(1, self.L+1):
            self.parameters['W' + str(l)] -= alpha*self.derivatives['dW' + str(l)]
            self.parameters['b' + str(l)] -= alpha*self.derivatives['db' + str(l)]
        
    def predict(self, x):
        self.forward_propagate(x)
        return self.parameters['a' + str(self.L)]
        
    def fit(self, X, Y, num_iter, alpha = 0.01):
        # print(Y.reshape(-1,1))
        for iter in range(0, num_iter):
            c = 0 #Stores the cost
            n_c = 0 #Stores the number of correct predictions
            
            for i in range(0, X.shape[0]):
              x = X[i].reshape((X[i].size, 1))
              y = Y[i].reshape(-1, 1) # convert shape from (10, ) to (10, 1)

              self.forward_propagate(x)
              self.compute_cost(y)
              self.compute_derivatives(y)
              self.update_parameters(alpha)

              c += self.parameters['C']

              y_pred = self.predict(x)
              #y_pred is the probability, so to convert it into a class value:
              #y_pred = (y_pred > 0.5) 
              # import pdb; pdb.set_trace()
              # print("y_pred:::",y_pred.shape)
              # print("Y:::",Y)
              max_prob = max(y_pred)
              max_prob_index = np.argmax(y_pred)
              result = np.where(max_prob_index == np.amax(max_prob_index))
              # result = result.reshape(-1,1)
              # listOfCordinates = list(zip(result[0], result[1]))
              # # travese over the list of cordinates
              # for cord in listOfCordinates:
              #   print(cord)
              y_pred = np.zeros(y_pred.shape[0])
              y_pred[max_prob_index] = 1
              # y_pred = y_pred[max_prob_index].reshape(-1, 1)
              # print(X.shape[0])

              # print("after zero::::",y_pred)
              # print("after yyyy::::",np.transpose(y))
              for i in np.transpose(y):
                if (y_pred == i).all():
                    n_c += 1
                # if (y_pred == i).all():
                #     n_c += 1


              
            c = c/X.shape[0]
            
            print('Iteration: ', iter)
            # print("cost:",c)
            print("Accuracy:", (n_c/X.shape[0])*100)

# %%
import pickle
#from PIL import Image
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# images = []
# class_index  = np.array([])

for i in range(0,5):
    fileName = "cifar-10-python/cifar-10-batches-py/data_batch_" + str(i+1)
    data = unpickle(fileName)
    class_names = unpickle("cifar-10-python/cifar-10-batches-py/batches.meta")
    data_key = "data".encode()
    label_key = "labels".encode()
    if(i == 0):
      images = data[data_key]
      class_index = data[label_key]
    else:
      images = np.append(images,(data[data_key]),axis=0)
      class_index = np.append(class_index,data[label_key],axis =0)
      classes = class_names[b'label_names']
    


def one_hot_encode(data):
    '''
    Description: Encode Target Label IDs to one hot vector of size L where L is the
    number of unique labels
    Params: data = list of label IDs
    Outputs: List of One Hot Vectors
    '''
    one_hot = np.zeros((len(data), 10))
    one_hot[np.arange(len(data)), data] = 1
    return one_hot

# %%
# print("class_index:::",class_index.shape)
X = images

y = one_hot_encode(class_index)

#Splitting the data into train set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.2)

#Defining the model architecture
architecture = np.array([3072,20,20,20,10])

#Creating the classifier
classifier = NeuralNetwork(architecture)

#Training the classifier
classifier.fit(X_train, y_train,100)

#Predicting the test set results:
n_c = 0
for i in range(0, X_test.shape[0]):
    x = X_test[i].reshape((X_test[i].size, 1))
    y = y_test[i]
    # print("y::",y)
    # import pdb; pdb.set_trace()
    y_pred = classifier.predict(x)
    # print("y_pred::",y_pred)
    max_prob = max(y_pred)
    max_prob_index = np.argmax(y_pred)
    y_pred = np.zeros(y_pred.shape[0])
    y_pred[max_prob_index] = 1
    # print(y_pred)
    # print(y)
    if (y_pred == y).all():
        n_c += 1


#print("Accuracy:", (n_c/X.shape[0])*100)
print("Total Accuracy:", (n_c/X_test.shape[0])*100 )

# # %%
# print(y_pred.shape)
# print(y.shape)

# # %%
# (classifier.parameters['a2'] - y[0].reshape(-1, 1)).shape

# # %%
# classifier.parameters['a2'].shape

# # %%
# y[0].shape

# # %%
# classifier.derivatives['dz2'].shape

# # %%
# y_pred

# # %%
