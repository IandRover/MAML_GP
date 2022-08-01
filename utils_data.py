import numpy
import jax.numpy as np

def load_data(bs=100):
    train_xs = numpy.load("./data/train_xs_100.npy")[:bs]
    train_xq = numpy.load("./data/train_xq_100.npy")[:bs]
    train_ys = numpy.load("./data/train_ys_100.npy")[:bs]
    train_yq = numpy.load("./data/train_yq_100.npy")[:bs]

    test_xs = numpy.load("./data/test_xs_100.npy")[:bs]
    test_xq = numpy.load("./data/test_xq_100.npy")[:bs]
    test_ys = numpy.load("./data/test_ys_100.npy")[:bs]
    test_yq = numpy.load("./data/test_yq_100.npy")[:bs]
    
    return train_xs, train_xq, train_ys, train_yq, test_xs, test_xq, test_ys, test_yq

main_dir = "./data/eva_data/miniimagenet/npy"
def load_more_data(bs=100):
    train_xs = numpy.concatenate([numpy.load(main_dir+f"/train_xs_100-{i}.npy") for i in range(3)], axis=0) 
    train_xq = numpy.concatenate([numpy.load(main_dir+f"/train_xq_100-{i}.npy") for i in range(3)], axis=0) 
    train_ys = numpy.concatenate([numpy.load(main_dir+f"/train_ys_100-{i}.npy") for i in range(3)], axis=0) 
    train_yq = numpy.concatenate([numpy.load(main_dir+f"/train_yq_100-{i}.npy") for i in range(3)], axis=0) 

    test_xs = numpy.load("./data/test_xs_100.npy")
    test_xq = numpy.load("./data/test_xq_100.npy")
    test_ys = numpy.load("./data/test_ys_100.npy")
    test_yq = numpy.load("./data/test_yq_100.npy")
    
    return train_xs, train_xq, train_ys, train_yq, test_xs, test_xq, test_ys, test_yq

def one_hot(a):
    b = numpy.zeros((a.size, a.max()+1))
    b[numpy.arange(a.size),a] = 1
    return b

def mask(Kernel, num_batch, na, nb): return np.multiply(Kernel, np.kron(np.eye(num_batch),np.ones((na,nb))))
