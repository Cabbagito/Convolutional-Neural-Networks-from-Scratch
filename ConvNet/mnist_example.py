import numpy as np
import ConvNetModule as net
from Optimizers import SGD, Momentum, RMSProp, Adam
from Layers import Conv, Max_Pooling, Flatten, Sigmoid, Linear, Relu, LeakyRelu, Output

X = np.load("MNIST/X_train.npy")
Y = np.load("MNIST/Y_train.npy")
X_test = np.load("MNIST/X_test.npy")
Y_test = np.load("MNIST/Y_test.npy")


def create_linear_model():

    model = net.CNN()
    model.add_layer(Linear(784, 500))
    model.add_layer(Relu())
    model.add_layer(Linear(500, 400))
    model.add_layer(Relu())
    model.add_layer(Linear(400, 300))
    model.add_layer(Relu())
    model.add_layer(Linear(300, 200))
    model.add_layer(Relu())
    model.add_layer(Linear(200, 100))
    model.add_layer(Relu())
    model.add_layer(Linear(100, 10))
    model.add_layer(Relu())
    model.add_layer(Output())
    model.set_optimizer(Adam(0.0001))
    return model


model = create_linear_model()
X = np.reshape(X, (X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]))
X_test = np.reshape(
    X_test, (X_test.shape[0], X_test.shape[1] * X_test.shape[2] * X_test.shape[3])
)
model.fit(
    X,
    Y,
    epochs=1,
    plot=True,
    batch_size=20,
    print_on_every=5,
    val_x=X_test,
    val_y=Y_test,
    val_size=20,
)


print(model.test_accuracy(X_test, Y_test))