def augment(Data, Labels, pct=0.3, merge=False):
    import numpy as np

    X = []
    Y = []
    translate = 7
    for i in range(Data.shape[0]):
        if np.random.rand() < pct:
            img = Data[i].copy()
            Y.append(Labels[i].copy())
            method = np.random.randint(0, 8)

            if method == 0:
                X.append(np.fliplr(img))
            elif method == 1:
                X.append(np.rot90(img))
            elif method == 2:
                X.append(np.rot90(img, k=3))
            elif method == 3:
                noise = np.random.rand(img.shape[0], img.shape[1], img.shape[2])
                noise -= 0.5
                noise *= 0.1
                X.append(img + noise)
            elif method == 4:  # up
                for collumn in range(img.shape[1]):
                    row = img[:, collumn, :]
                    for i in range(row.shape[0]):
                        if i >= row.shape[0] - translate:
                            row[i, :] = 0
                        else:
                            row[i, :] = row[i + translate, :]
                X.append(img)
            elif method == 5:  # down
                for collumn in range(img.shape[1]):
                    row = img[:, collumn, :]
                    for i in reversed(range(row.shape[0])):
                        if i <= translate:
                            row[i, :] = 0
                        else:
                            row[i, :] = row[i - translate, :]
                X.append(img)
            elif method == 6:  # right
                for row in range(img.shape[0]):
                    collumn = img[row, :, :]
                    for i in reversed(range(collumn.shape[0])):
                        if i <= translate:
                            collumn[i, :] = 0
                        else:
                            collumn[i, :] = collumn[i - translate, :]
                X.append(img)
            elif method == 7:  # left
                for row in range(img.shape[0]):
                    collumn = img[row, :, :]
                    for i in range(collumn.shape[0]):
                        if i >= collumn.shape[0] - translate:
                            collumn[i, :] = 0
                        else:
                            collumn[i, :] = collumn[i + translate, :]
                X.append(img)

    X = np.array(X)
    Y = np.array(Y)
    if merge:
        X = np.append(Data, X, axis=0)
        Y = np.append(Labels, Y, axis=0)

    return X, Y


def padding(x, kernel_size, stride):
    import numpy as np

    x_h = x.shape[1]
    x_w = x.shape[2]
    p_h = ((x_h - 1) * stride[0] - x_h + kernel_size[1]) / 2
    p_w = ((x_w - 1) * stride[1] - x_w + kernel_size[0]) / 2
    p_h = int(p_h)
    p_w = int(p_w)
    retvalue = np.empty(
        (x.shape[0], x_h + 2 * p_h, x_w + 2 * p_w, x.shape[3]), dtype=np.float32
    )

    for img in range(x.shape[0]):
        retvalue[img] = np.pad(x[img], ((p_w, p_w), (p_h, p_h), (0, 0)))
    return retvalue


def calc_img_size_from_params(img_shape, kernel_shape, stride):
    height = (img_shape[1] - kernel_shape[0]) / stride[0] + 1
    width = height = (img_shape[2] - kernel_shape[1]) / stride[1] + 1
    return (int(height), int(width))


def shuffle(X, Y):
    from random import shuffle
    from numpy import array

    data = list(zip(X, Y))
    shuffle(data)
    X_ret, Y_ret = zip(*data)
    X = array(X)
    Y = array(Y)
    return X, Y


def select_n(X, Y, n):
    X, Y = shuffle(X, Y)
    return X[:n], Y[:n]
