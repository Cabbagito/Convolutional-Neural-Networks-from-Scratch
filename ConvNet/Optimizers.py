class Adam:
    def __init__(self, learning_rate=0.0001, p1=0.9, p2=0.999):
        self.lr = learning_rate
        self.p1 = p1
        self.p2 = p2
        self.type = "Adam"
        self.__initialized = False
        self.t = 0
        self.r = {}
        self.s = {}

    def step(self, parameters, gradients):
        init_done = True
        from numpy import sqrt

        self.t += 1
        for i in parameters:
            if not self.__initialized:
                from numpy import zeros_like

                init_done = False
                self.r[i] = zeros_like(parameters[i])
                self.s[i] = zeros_like(parameters[i])

            self.s[i] = self.p1 * self.s[i] + (1 - self.p1) * gradients[i]
            self.r[i] = (
                self.p2 * self.r[i] + (1 - self.p2) * gradients[i] * gradients[i]
            )
            s_corr = self.s[i] / (1 - self.p1 ** self.t)
            r_corr = self.r[i] / (1 - self.p2 ** self.t)
            parameters[i] -= self.lr * s_corr / (sqrt(r_corr) + 1e-8)

        if init_done == False:
            self.__initialized = True
        return parameters


class SGD:
    def __init__(self, learning_rate=0.0001):
        self.lr = learning_rate
        self.type = "SGD"

    def step(self, parameters, gradients):
        for i in parameters:
            parameters[i] -= self.lr * gradients[i]
        return parameters


class Momentum:
    def __init__(self, learning_rate=0.0001, omega=0.9):
        self.v = {}
        self.omega = omega
        self.lr = learning_rate
        self.__initialized = False
        self.type = "Momentum"

    def step(self, parameters, gradients):
        init_done = True
        for i in parameters:
            if not self.__initialized:
                init_done = False
                from numpy import zeros, float32

                self.v[i] = zeros(parameters[i].shape, dtype=float32)

            self.v[i] = self.omega * self.v[i] - self.lr * gradients[i]
            parameters[i] += self.v[i]

        if init_done == False:
            self.__initialized = True

        return parameters


class RMSProp:
    def __init__(self, learning_rate=0.0001, decay_rate=0.9):
        self.lr = learning_rate
        self.p = decay_rate
        self.type = "RMSProp"
        self.r = {}
        self.__initialized = False

    def step(self, parameters, gradients):
        from numpy import sqrt

        init_done = True
        for i in parameters:
            if not self.__initialized:
                from numpy import zeros_like, float32

                self.r[i] = zeros_like(parameters[i], dtype=float32)
                init_done = False

            self.r[i] = self.p * self.r[i] + (1 - self.p) * gradients[i] * gradients[i]
            parameters[i] -= self.lr / (sqrt(1e-6 + self.r[i])) * gradients[i]

        if init_done == False:
            self.__initialized = True
        return parameters
