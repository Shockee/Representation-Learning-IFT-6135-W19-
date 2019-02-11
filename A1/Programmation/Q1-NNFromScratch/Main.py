from NN import NN

def main():
    MLP = NN((600,600), 784, 10, data_path='mnist.pkl.npy')
    MLP.train(10)

    return None




if __name__ == '__main__':
    main()