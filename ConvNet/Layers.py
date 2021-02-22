import numpy as np
import Preprocessing


class Conv:
    def __init__(
        self,
        n_input_filters,
        n_filters,
        kernel_shape=(2, 2),
        stride=(1, 1),
        padding="valid",
        xavier=True,
    ):
        self.n_filters = n_filters
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.padding = padding
        self.__generate_params(n_input_filters)
        self.dW = None
        self.type = "Conv"

    def operation(self, x):  # x = (n,h,w,c)  #W = (n_f,h,w,c)

        if self.padding == "same":
            x = Preprocessing.padding(x, self.kernel_shape, self.stride)

        size = Preprocessing.calc_img_size_from_params(
            x.shape, self.kernel_shape, self.stride
        )
        output = np.empty(
            (x.shape[0], size[0], size[1], self.n_filters), dtype=np.float32
        )
        for img in range(x.shape[0]):
            for kernel in range(self.n_filters):
                output[img, :, :, kernel] = self.__conv(
                    x[img], self.W[kernel], self.stride, size
                )
        return output

    def __conv(self, img, kernel, stride, expected_size):
        retval = np.empty((expected_size[0], expected_size[1]), dtype=np.float32)
        for i in range(expected_size[0]):
            for j in range(expected_size[1]):
                retval[i, j] = np.sum(
                    np.multiply(
                        img[
                            i * stride[0] : i * stride[0] + kernel.shape[1],
                            j * stride[1] : j * stride[1] + kernel.shape[0],
                            :,
                        ],
                        kernel,
                    )
                )

        return retval.copy()

    def __generate_params(self, n_input_filters, xavier=False):
        self.W = np.random.randn(
            self.n_filters, self.kernel_shape[0], self.kernel_shape[1], n_input_filters
        )
        if xavier:
            self.W *= np.sqrt(2 / self.n_filters)

    def gradient(self, x, grad):
        if self.padding == "same":
            x = Preprocessing.padding(
                x, (self.W.shape[1], self.W.shape[2]), self.stride
            )

        self.dW = np.zeros_like(self.W, dtype=np.float32)
        retval = np.zeros_like(x, dtype=np.float32)

        grad_pad = np.pad(
            grad, ((0,), (self.W.shape[2] - 1,), (self.W.shape[1] - 1,), (0,))
        )
        W_180 = np.zeros_like(self.W, dtype=np.float32)
        for i in range(W_180.shape[1]):
            for j in range(W_180.shape[2]):
                W_180[:, i, j, :] = self.W[
                    :, self.W.shape[1] - i - 1, self.W.shape[2] - j - 1, :
                ]

        for img in range(x.shape[0]):
            for f in range(self.dW.shape[0]):
                for c in range(self.dW.shape[3]):
                    for m in range(grad.shape[1]):
                        for n in range(grad.shape[2]):
                            for i in range(self.dW.shape[1]):
                                for j in range(self.dW.shape[2]):
                                    self.dW[f, i, j, c] += (
                                        grad[img, m, n, f]
                                        * x[
                                            img,
                                            i * self.stride[0] + m,
                                            j * self.stride[1] + n,
                                            c,
                                        ]
                                    )
                    for i in range(x.shape[1]):
                        for j in range(x.shape[2]):
                            for m in range(self.W.shape[1]):
                                for n in range(self.W.shape[2]):

                                    retval[img, i, j, c] += (
                                        grad_pad[
                                            img,
                                            i + m,
                                            j + n,
                                            f,
                                        ]
                                        * W_180[f, m, n, c]
                                    )

        return retval


class Max_Pooling:
    def __init__(self, shape=(2, 2), stride=(2, 2)):
        self.shape = shape
        self.stride = stride
        self.type = "Max_Pooling"

    def operation(self, x):
        self.__argmaxes = []
        self.__grad_spots = []

        size = Preprocessing.calc_img_size_from_params(x.shape, self.shape, self.stride)

        output = np.empty((x.shape[0], size[0], size[1], x.shape[3]), dtype=np.float32)

        for img in range(x.shape[0]):
            output[img], argmaxes, grad_spots = self.__pool(x[img], size)
            self.__argmaxes.append(argmaxes)
            self.__grad_spots.append(grad_spots)
        self.__argmaxes = np.array(self.__argmaxes)
        self.__grad_spots = np.array(self.__grad_spots)
        return output

    def __pool(self, x, expected_size):
        retval = np.empty(
            (expected_size[0], expected_size[1], x.shape[2]), dtype=np.float32
        )
        argmaxes = []
        grad_spots = []
        s = list(self.stride)
        for i in range(expected_size[0]):
            for j in range(expected_size[1]):
                for c in range(x.shape[2]):
                    max_index = np.argmax(
                        x[
                            i * self.stride[0] : i * self.stride[0] + self.shape[1],
                            j * self.stride[1] : j * self.stride[1] + self.shape[0],
                            c,
                        ]
                    )
                    max_index = np.unravel_index(max_index, self.shape)
                    max_index = list(max_index)
                    max_index[0] += i * s[0]
                    max_index[1] += j * s[1]
                    max_index.append(c)
                    argmaxes.append(max_index)
                    grad_spots.append([i, j, c])
                    retval[i, j, c] = np.max(
                        x[
                            i * self.stride[0] : i * self.stride[0] + self.shape[1],
                            j * self.stride[1] : j * self.stride[1] + self.shape[0],
                            c,
                        ]
                    )
        return retval.copy(), argmaxes, grad_spots

    def gradient(self, x, grad):
        retval = np.zeros_like(x, dtype=np.float32)
        for img in range(x.shape[0]):
            for region in range(self.__argmaxes.shape[1]):
                retval[
                    img,
                    self.__argmaxes[img, region, 0],
                    self.__argmaxes[img, region, 1],
                    self.__argmaxes[img, region, 2],
                ] = grad[
                    img,
                    self.__grad_spots[img, region, 0],
                    self.__grad_spots[img, region, 1],
                    self.__grad_spots[img, region, 2],
                ]
        return retval


class Relu:
    def __init__(self):
        self.type = "Relu"

    def operation(self, x):
        return x * (x > 0)

    def gradient(self, x, grad):
        return (x > 0) * grad


class LeakyRelu:
    def __init__(self, coef=0.01):
        self.type = "LeakyRelu"
        self.coef = coef

    def operation(self, x):
        x1 = x * (x > 0)
        x2 = (x <= 0) * x * self.coef
        return x1 + x2

    def gradient(self, x, grad):
        x1 = (x > 0) * grad
        x2 = (x <= 0) * grad * self.coef
        return x1 + x2


class Flatten:
    def __init__(self):
        self.type = "Flatten"

    def operation(self, x):
        return np.reshape(x, (x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]))

    def gradient(self, x, grad):
        return np.reshape(grad, x.shape)


class Linear:
    def __init__(self, input_size, output_size, xavier=True):
        self.__generate_params(input_size, output_size, xavier)
        self.dW = None
        self.db = None
        self.type = "Linear"

    def __generate_params(self, input_size, output_size, xavier):
        self.W = np.random.randn(input_size, output_size)
        self.b = np.random.randn(1, output_size)
        if xavier:
            self.W *= 2 / np.sqrt(input_size + output_size)

    def operation(self, X):
        return np.dot(X, self.W) + self.b

    def gradient(self, x, grad):
        self.dW = np.dot(x.T, grad)
        self.db = np.average(grad, axis=0)
        self.db = np.reshape(self.db, (1, self.db.shape[0]))
        return np.dot(grad, self.W.T)


class Sigmoid:
    def __init__(self):
        self.activation = None
        self.type = "Sigmoid"

    def operation(self, X):
        self.activation = 1 / (1 + np.exp(-X))
        return self.activation

    def gradient(self, x, grad):
        return (self.activation * (1 - self.activation)) * grad


class Output:
    def __init__(self):
        self.type = "Output"

    def operation(self, X):
        retval = np.empty(X.shape, dtype=np.float32)
        sums = np.sum(np.exp(X), axis=1)
        for i in range(X.shape[1]):
            retval[:, i] = np.exp(X[:, i]) / sums
        return retval
