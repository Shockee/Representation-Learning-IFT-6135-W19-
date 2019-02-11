from NN import NN
import matplotlib.pyplot as plt

def main():
    loss_init = []

    MLP = NN(784, 10, data_path='mnist.pkl.npy', hidden_dims=(600,600), activation_type='tanh')

    #loss_init.append(MLP.train(10, 'zeros'))
    loss_init.append(MLP.train(10, 'normal_dist'))
    #loss_init.append(MLP.train(10,'glorot'))

    #plt.p#lot(loss_init[0], label='Zeros')
    #plt.plot(loss_init[1], label='Dist. Normale')
    #plt.plot(loss_init[2], label='Glorot')
    #plt.legend(loc='upper left')

    plt.show()
    return None




if __name__ == '__main__':
    main()