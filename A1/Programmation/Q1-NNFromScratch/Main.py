from NN import NN
import random
import matplotlib.pyplot as plt


def main():

    compare_weight_initialization_methods()
    # random_search(10)
    return None


def compare_weight_initialization_methods():
    loss_init = []

    mlp = NN(784, 10, data_path='mnist.pkl.npy', hidden_dims=(600, 600), activation_type='relu')

    loss_init.append(mlp.train(10, 'zeros'))
    loss_init.append(mlp.train(10, 'normal_dist'))
    loss_init.append(mlp.train(10,'glorot'))

    plt.plot(loss_init[0], label='Zeros')
    plt.plot(loss_init[1], label='Dist. Normale')
    plt.plot(loss_init[2], label='Glorot')
    plt.legend(loc='upper left')

    plt.show()


def random_search(num_search):
    activations = ['relu', 'sigmoid', 'tanh']

    for i in range(num_search):
        # Random initialization of hyper-parameters
        dim_l1 = random.randint(450, 650)
        dim_l2 = random.randint(450, 650)
        activation = activations[random.randint(0, 2)]
        learning_rate = random.uniform(0.001, 0.05)

        batch_random = random.randint(4, 8)
        mini_batch_size = 2**batch_random

        # Creation and training of the NN
        mlp = NN(784, 10, data_path='mnist.pkl.npy', hidden_dims=(dim_l1, dim_l2), activation_type=activation,
                 batch_size=mini_batch_size, learning_rate=learning_rate)

        print("Dim:("+ str(dim_l1) + "," + str(dim_l2) + "), activation: " + activation + " , learning rate: "
              + str(learning_rate) + ", batch_size = " + str(mini_batch_size))

        mlp.train(10, 'glorot')


if __name__ == '__main__':
    main()
