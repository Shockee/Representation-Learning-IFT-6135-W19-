import numpy as np
import pickle as pickle
import matplotlib.pyplot as plt

class NN (object):
    learning_rate = 0.00001
    data_path = ''
    hidden_dims = None
    num_hidden = None
    input_size = 0
    num_class = 0
    W = []
    b = []
    grad = []
    a1 = []
    a2 = []
    o = []
    l_vector = []
    plt_vector = []
    batch_size = 0

    def __init__(self, hidden_dims, input_size, num_class, data_path):
        self.data_path = data_path
        self.hidden_dims = hidden_dims
        self.input_size = input_size
        self.num_class = num_class
        self.num_hidden = len(hidden_dims)

    def initialize_weights(self, input_size, num_class, init_type='normal_dist'):
        if init_type == 'zeros':

            # W is a vector that holds w matrices for each hidden layer and for the output layer
            w1 = np.zeros((input_size, self.hidden_dims[0]))
            w2 = np.zeros((self.hidden_dims[0], self.hidden_dims[1]))
            o = np.zeros((self.hidden_dims[1], num_class))
            b1 = np.zeros((1,self.hidden_dims[0]))
            b2 = np.zeros((1,self.hidden_dims[1]))
            b3 = np.zeros((1,num_class))


        if init_type == 'normal_dist':
            w1 = np.random.normal(0, 0.1, (input_size, self.hidden_dims[0]))
            w2 = np.random.normal(0, 0.1, (self.hidden_dims[0], self.hidden_dims[1]))
            o = np.random.normal(0, 0.1, (self.hidden_dims[1], num_class))
            b1 = np.zeros((1,self.hidden_dims[0]))
            b2 = np.zeros((1,self.hidden_dims[1]))
            b3 = np.zeros((1,num_class))

        self.W = [w1, w2, o]
        self.b = [b1,b2,b3]

    def forward(self, input):
        try:
            #First logit
            h1 = input.dot(self.W[0]) + self.b[0]

            #First non-linearity
            a1 = self.activation(h1)

            #Second Logit
            h2 = np.dot(a1, self.W[1]) + self.b[1]

            #Second linearity
            a2 = self.activation(h2)

            #Output logit
            o = np.dot(a2, self.W[2]) + self.b[2]

            #Saving activations for backprop
            self.a1 = a1
            self.a2 = a2
            self.o = o

            #Return the softmax
            return self.softmax(o)
        except:
            t =2


    def activation(self, input, type='relu'):
        if type == 'relu':
            return np.maximum(input, 0, input)

    def relu_derivative(self,x):
        return 1. * (x > 0)

    def loss(self, pred_vector, correct_label, type='cross-entropy'):
        if type == 'cross-entropy':
            return np.average(-np.log(pred_vector[range(self.batch_size), correct_label]))

    def softmax(self, input):
        #Stabilize the softmax
        input = input - np.max(input)

        #Computes the softmax function
        exp_score = np.exp(input)
        return exp_score/np.sum(exp_score, axis=1, keepdims=True)

    def backward(self, input, correct_label, prediction, mini_batch_size):
        #Calculate the loss
        loss = self.loss(prediction,correct_label)
        if(len(self.l_vector) == 100):
            print(np.average(self.l_vector))
            avg = np.average(self.l_vector)
            self.plt_vector.append(avg)
            self.l_vector = []

        self.l_vector.append(loss)

        #Derivative of the output layer ( np.multiply(y, np.multiply(1-y, targets-y)))
        do = prediction
        do[range(self.batch_size), correct_label] -= 1
        dwo = np.dot(self.a2.T, do)
        bo = np.sum(do,axis=0)

        #Derivative of the second layer
        dh2 = np.dot(do, self.W[2].T) * self.relu_derivative(self.a2)
        dw2 = np.dot(self.a1.T, dh2)
        b2 = np.sum(dh2,axis=0)

        #Derivative of the first layer
        dh1 = np.dot(dh2, self.W[1].T) * self.relu_derivative(self.a1)
        dw1 = np.dot(input.T, dh1)
        b1 = np.sum(dh1,axis=0)

        return [dw1,dw2,dwo], [b1,b2,bo]

    def update(self, grads, gbias):

        #Update ouput layer
        self.W[2] = self.W[2] - self.learning_rate * grads[2]
        self.b[2] = self.b[2] - self.learning_rate * gbias[2]

        #Update last hidden layer
        self.W[1] = self.W[1] - self.learning_rate * grads[1]
        self.b[1] = self.b[1] - self.learning_rate * gbias[1]

        #Update first hidden layer
        self.W[0] = self.W[0] - self.learning_rate * grads[0]
        self.b[0] = self.b[0] - self.learning_rate * gbias[0]

    def train(self, epochs):

        #Manual setting of hyperparameters
        self.batch_size = 32

        #Data retrieval
        data = np.load(self.data_path)[0]
        validation = np.load(self.data_path)[1]

        #Weight initialization
        self.initialize_weights(self.input_size, self.num_class)

        #Training
        for epoch in range(0,epochs):
            #Reinitialize batchsize
            self.batch_size = 32
            for i in range(0,len(data[0]), self.batch_size):

                batch_end_index = i + self.batch_size

                #Case where batchsize would cause overflow the data
                if batch_end_index > len(data[0]):
                    batch_end_index = len(data[0])
                    self.batch_size = len(data[0]) - i

                #Initialize current training batch
                x = data[0][range(i, batch_end_index)]
                t = data[1][range(i, batch_end_index)]

                #Foward pass
                prediction = self.forward(x)

                #Backprop
                grads, gbias = self.backward(x,t,prediction,self.batch_size)

                #Weight update
                self.update(grads, gbias)

        #self.test(data)
        self.test(validation)
        plt.plot(self.plt_vector)
        plt.show()
        return None

    def test(self, validation_set):
        results = np.sum(np.equal(np.argmax(self.forward(validation_set[0]), axis=1),validation_set[1]).astype(int))

        print (results/len(validation_set[0]))

