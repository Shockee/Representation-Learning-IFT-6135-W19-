import numpy as np
import pickle as pickle
import matplotlib.pyplot as plt

class NN (object):

    # Network initialization parameters
    data_path = ''
    num_hidden = None
    input_size = 0
    num_class = 0

    # Network hyperparameters
    hidden_dims = None
    learning_rate = 0.00001
    activation_type = ''
    batch_size = 0

    # Network parameters
    params = {
        'w': [],
        'b': []
    }

    # Cache for backpropagation
    cache = []

    # Utils for loss
    l_vector = []
    plt_vector = []


    def __init__(self, input_size, num_class, data_path, learning_rate=0.00001, hidden_dims=(600, 600), batch_size=32,
                 activation_type = 'relu'):

        self.data_path = data_path
        self.hidden_dims = hidden_dims
        self.input_size = input_size
        self.num_class = num_class
        self.learning_rate = learning_rate
        self.num_hidden = len(hidden_dims)
        self.batch_size = batch_size
        self.activation_type = activation_type

    def initialize_weights(self, input_size, num_class, init_type):
        #Zero initialization
        if init_type == 'zeros':
            self.params['w'].append(np.zeros((input_size, self.hidden_dims[0])))
            self.params['w'].append(np.zeros((self.hidden_dims[0], self.hidden_dims[1])))
            self.params['w'].append(np.zeros((self.hidden_dims[1], num_class)))
            self.params['b'].append(np.zeros((1,self.hidden_dims[0])))
            self.params['b'].append(np.zeros((1,self.hidden_dims[1])))
            self.params['b'].append(np.zeros((1,num_class)))

        #Normal distribution initialization(with biases 0)
        if init_type == 'normal_dist':
           self.params['w'].append(np.random.normal(0, 0.1, (input_size, self.hidden_dims[0])))
           self.params['w'].append(np.random.normal(0, 0.1, (self.hidden_dims[0], self.hidden_dims[1])))
           self.params['w'].append(np.random.normal(0, 0.1, (self.hidden_dims[1], num_class)))
           self.params['b'].append(np.zeros((1,self.hidden_dims[0])))
           self.params['b'].append(np.zeros((1,self.hidden_dims[1])))
           self.params['b'].append(np.zeros((1,num_class)))

        #Glorot initialization
        if init_type == 'glorot':
            d1 = np.sqrt(6./(input_size + self.hidden_dims[0]))
            d2 = np.sqrt(6./(self.hidden_dims[0] + self.hidden_dims[1]))
            do = np.sqrt(6./(self.hidden_dims[1] + num_class))

            self.params['w'].append(np.random.uniform(-d1, d1, (input_size, self.hidden_dims[0])))
            self.params['w'].append(np.random.uniform(-d2, d2, (self.hidden_dims[0], self.hidden_dims[1])))
            self.params['w'].append(np.random.uniform(-do, do, (self.hidden_dims[1], num_class)))
            self.params['b'].append(np.zeros((1,self.hidden_dims[0])))
            self.params['b'].append(np.zeros((1,self.hidden_dims[1])))
            self.params['b'].append(np.zeros((1,num_class)))

    def forward(self, input):
        #First logit
        h1 = input.dot(self.params['w'][0]) + self.params['b'][0]

        #First non-linearity
        a1 = self.activation(h1)

        #Second Logit
        h2 = np.dot(a1, self.params['w'][1]) + self.params['b'][1]

        #Second non-linearity
        a2 = self.activation(h2)

        #Output logit
        o = np.dot(a2, self.params['w'][2]) + self.params['b'][2]

        #Caching activations for backprop
        self.cache.append(a1)
        self.cache.append(a2)

        #Return the softmax
        return self.softmax(o)


    def activation(self, input):
        if self.activation_type == 'relu':
            return np.maximum(input, 0, input)

        if self.activation_type == 'sigmoid':
            return 1 / (np.exp(-input) + 1)

        if self.activation_type == 'tanh':
            return (np.exp(input) - np.exp(-input)) / (np.exp(input) + np.exp(-input))

    def activation_derivative(self, x, num_layer):
        if self.activation_type == 'relu':
            return 1. * (x > 0)

        if self.activation_type == 'sigmoid':
            return self.activation(x)*(1 - self.activation(x))

        if self.activation_type == 'tanh':
            return 1 - np.square(self.activation(x))


    def loss(self, pred_vector, correct_label, batch_size, type='cross-entropy'):
        if type == 'cross-entropy':
            return np.average(-np.log(pred_vector[range(batch_size), correct_label]))

    def softmax(self, input):
        #Stabilize the softmax
        input = input - np.max(input)

        #Computes the softmax function
        exp_score = np.exp(input)
        return exp_score/np.sum(exp_score, axis=1, keepdims=True)

    def backward(self, input, correct_label, prediction, mini_batch_size):
        #Calculate the loss
        loss = self.loss(prediction,correct_label, mini_batch_size)
        self.l_vector.append(loss)

        #Derivative of the output layer ( np.multiply(y, np.multiply(1-y, targets-y)))
        do = prediction
        do[range(mini_batch_size), correct_label] -= 1
        dwo = np.dot(self.cache[1].T, do)
        bo = np.sum(do,axis=0)

        #Derivative of the second layer
        dh2 = np.dot(do, self.params['w'][2].T) * self.activation_derivative(self.cache[1], 2)
        dw2 = np.dot(self.cache[0].T, dh2)
        b2 = np.sum(dh2,axis=0)

        #Derivative of the first layer
        dh1 = np.dot(dh2, self.params['w'][1].T) * self.activation_derivative(self.cache[0], 1)
        dw1 = np.dot(input.T, dh1)
        b1 = np.sum(dh1,axis=0)

        return [dw1,dw2,dwo], [b1,b2,bo]

    def update(self, grads, gbias):

        #Update ouput layer
        self.params['w'][2] = self.params['w'][2] - self.learning_rate * grads[2]
        self.params['b'][2] =  self.params['b'][2] - self.learning_rate * gbias[2]

        #Update last hidden layer
        self.params['w'][1] = self.params['w'][1] - self.learning_rate * grads[1]
        self.params['b'][1] =  self.params['b'][1] - self.learning_rate * gbias[1]

        #Update first hidden layer
        self.params['w'][0] = self.params['w'][0] - self.learning_rate * grads[0]
        self.params['b'][0] =  self.params['b'][0] - self.learning_rate * gbias[0]

    def train(self, epochs, weight_init_method):
        loss_history = []

        #Data retrieval
        data = np.load(self.data_path)[0]
        validation = np.load(self.data_path)[1]

        #Weight initialization
        self.initialize_weights(self.input_size, self.num_class, weight_init_method)

        #Training
        for epoch in range(0,epochs):
            for i in range(0,len(data[0]), self.batch_size):
                current_batch_size = self.batch_size
                batch_end_index = i + self.batch_size

                #Case where batchsize would cause overflow the data
                if batch_end_index > len(data[0]):
                    batch_end_index = len(data[0])
                    current_batch_size = len(data[0]) - i

                #Initialize current training batch
                x = data[0][range(i, batch_end_index)]
                t = data[1][range(i, batch_end_index)]

                #Foward pass
                prediction = self.forward(x)

                #Backprop
                grads, gbias = self.backward(x,t,prediction,current_batch_size)

                #Weight update
                self.update(grads, gbias)

                #Cache clearing
                self.cache = []

            loss_history.append(np.average(self.l_vector))
            print(np.average(self.l_vector))
            self.l_vector = []

        self.test(validation)
        return loss_history

    def test(self, validation_set):
        results = np.sum(np.equal(np.argmax(self.forward(validation_set[0]), axis=1),validation_set[1]).astype(int))

        print(results/len(validation_set[0]))

