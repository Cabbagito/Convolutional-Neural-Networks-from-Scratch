import Preprocessing
import Optimizers
import Layers


class CNN:
    def __init__(self):
        self.values = {}
        self.gradients = {}
        self.optimizer = None
        self.model_layers = []
        self.l2 = False
        self.l2_coef = 0.0001

    def add_layer(self, layer):
        self.model_layers.append(layer)

    def forward(self, X):
        self.values["X0"] = X
        for i in range(len(self.model_layers)):
            X = self.model_layers[i].operation(X)
            self.values["X" + str(i + 1)] = X
        return X

    def backward(self, Y):
        self.gradients["X" + str(len(self.model_layers) - 1)] = self.__grad_from_loss(
            Y, self.values["X" + str(len(self.model_layers))]
        )
        for i in reversed(range(len(self.model_layers) - 1)):
            self.gradients["X" + str(i)] = self.model_layers[i].gradient(
                self.values["X" + str(i)], self.gradients["X" + str(i + 1)]
            )

    def fit(
        self,
        X,
        Y,
        epochs,
        batch_size=None,
        val_x=None,
        val_y=None,
        val_size=None,
        plot=False,
        print_on_every=10,
    ):
        if batch_size == None:
            batch_size = X.shape[0]
        from numpy import array

        val_x = array(val_x)
        losses = []
        val_losses = []
        val_losses_text = ""
        validating = not (val_x.any() == None)
        if validating:
            if val_size == None:
                val_size = val_x.shape[0]

        from tqdm import tqdm

        n_iter = int(X.shape[0] / batch_size)

        for epoch in range(epochs):
            print("Epoch " + str(epoch + 1))
            iterator = tqdm(range(n_iter), desc="Begining Training")
            for batch in iterator:
                batch_x = X[batch * batch_size : (batch + 1) * batch_size]
                batch_y = Y[batch * batch_size : (batch + 1) * batch_size]

                yhat = self.forward(batch_x)
                self.backward(batch_y)

                loss = self.loss(batch_y, yhat)
                losses.append(loss)

                self.apply_gradients()

                if self.l2:
                    self.__apply_l2()

                if validating:
                    val_x_batch, val_y_batch = Preprocessing.select_n(
                        val_x, val_y, val_size
                    )
                    o = self.forward(val_x_batch)
                    val_loss = self.loss(val_y_batch, o)
                    val_losses.append(val_loss)
                    val_losses_text = "  Validation Loss: " + str(val_loss)

                if batch % print_on_every == 0:

                    iterator.set_description("Loss: " + str(loss) + val_losses_text)

        if plot == True:
            import matplotlib.pyplot as plt

            plt.plot(losses)
            if validating:
                plt.plot(val_losses)
            plt.legend(["Loss", "Validation Loss"])
            plt.title("Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.show()

    def predict(self, x):
        from numpy import argmax

        return argmax(self.forward(x), axis=1)

    def test_accuracy(self, x, y):
        from numpy import argmax, sum

        predictions = self.predict(x)
        correct = predictions == argmax(y, axis=1)
        return sum(correct) / x.shape[0]

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def apply_gradients(self):
        parameters, gradients = self.get_weights()
        self.set_weights(self.optimizer.step(parameters, gradients))

    def loss(self, y, yhat):
        from numpy import average, sum, log

        return average(-sum(y * log(yhat), axis=1))

    def __grad_from_loss(self, y, yhat):
        return yhat - y

    def __apply_l2(self):
        parameters, _ = self.get_weights()

        for param in parameters:
            parameters[param] -= self.l2_coef * 2 * parameters[param]
        self.set_weights(parameters)

    def get_weights(self):
        params = {}
        gradients = {}
        for l in range(len(self.model_layers)):
            if self.model_layers[l].type == "Linear":
                params["W" + str(l + 1)] = self.model_layers[l].W
                params["b" + str(l + 1)] = self.model_layers[l].b
                gradients["W" + str(l + 1)] = self.model_layers[l].dW
                gradients["b" + str(l + 1)] = self.model_layers[l].db
            elif self.model_layers[l].type == "Conv":
                params["W" + str(l + 1)] = self.model_layers[l].W
                gradients["W" + str(l + 1)] = self.model_layers[l].dW
        return params, gradients

    def set_weights(self, weights):
        for l in range(len(self.model_layers)):
            if self.model_layers[l].type == "Linear":
                self.model_layers[l].W = weights["W" + str(l + 1)]
                self.model_layers[l].b = weights["b" + str(l + 1)]
            elif self.model_layers[l].type == "Conv":
                self.model_layers[l].W = weights["W" + str(l + 1)]
