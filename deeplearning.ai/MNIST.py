import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os

data = pd.read_csv("mnist_train.csv", sep=",")  # 60000 samples
data2 = pd.read_csv("mnist_test2.csv", sep=",")  # 10000 samples

print(data.head())
print(data.info())


########################################################################################################################

def plot1(cost_record):
    Y = cost_record[ :, 1]
    X = cost_record[ :, 2]

    plt.scatter(X, Y)
    plt.xlabel('batch')
    plt.ylabel('cost')
    plt.show()


########################################################################################################################

def softmax(raw_preds):
    out = np.exp(raw_preds)
    return out / np.sum(out)

########################################################################################################################

def softmaxall(P3,parameters,b):
    probs_all = []

    (m, height, width, nf2) = P3.shape
    fulldpool=np.zeros(shape=(0,height, width, nf2))
    w3 = parameters["W3"]
    w4 = parameters["W4"]
    b3 = parameters["b3"]
    b4 = parameters["b4"]
    for i in range(0,m):

        fc = P3[i, :, :, :].reshape(( nf2 * height * width, 1))  # partial flatten pooled layer
        z = w3.dot(fc) + b3  # first dense layer
        z[z <= 0] = 0  # pass through ReLU non-linearity

        out = w4.dot(z) + b4  # second dense layer
        prob=softmax(out)
        probs_all.append(softmax(out).reshape(1,10))
        dout = prob - b[i].reshape(10,1)  # derivative of loss w.r.t. final dense layer output
        dw4 = dout.dot(z.T)  # loss gradient of final dense layer weights
        db4 = np.sum(dout, axis=1).reshape(b4.shape)  # loss gradient of final dense layer biases
        dz = w4.T.dot(dout)  # loss gradient of first dense layer outputs
        dz[z <= 0] = 0  # backpropagate through ReLU
        dw3 = dz.dot(fc.T)
        db3 = np.sum(dz, axis=1).reshape(b3.shape)
        dfc = w3.T.dot(dz)  # loss gradients of fully-connected layer (pooling layer)

        dpool = dfc.reshape(P3[i,:,:,:].shape)  # reshape fully connected into dimensions of pooling layer
        fulldpool=np.append(fulldpool,dpool[np.newaxis,:,:,:],axis=0)
        print("fulldpool shape is", fulldpool.shape)
    probs_all=np.array(probs_all)
    print("probs_all shape is",probs_all.shape)
    return probs_all,dw3,dw4,db3,db4,fulldpool

########################################################################################################################

def safe_ln(x, minval=0.0000000001):  # avoid divide by zero
    return np.log(x.clip(min=minval))


########################################################################################################################

X_train = data.drop(['label'], axis=1)  # df1 = df1.drop(['B', 'C'], axis=1)  axis =1 means along row
y_train = data[['label']]  # df1 = df[['a','d']]
X_test = data2.drop(['label'], axis=1)
y_test = data2[['label']]

print("X_train head")
print(X_train.head())
print("y_train head")
print(y_train.head())

np_X_train = X_train.as_matrix()  # convert pandas to numpy
np_y_train = y_train.as_matrix()
np_X_test = X_test.as_matrix()
np_y_test = y_test.as_matrix()


########################################################################################################################


def initializeWeight(size):
    return np.random.standard_normal(size=size) * 0.01


########################################################################################################################

def display(n):
    print("\nMNIST image\n")

    d = np_X_train[n]
    d = d.reshape(28, 28)
    for row in range(0, 28):
        for col in range(0, 28):
            print("%02X " % d[row][col], end="")
        print("")

    lbl = np_y_train[n]
    print("\ndigit = ", lbl)

    plt.imshow(d, cmap=plt.get_cmap('gray_r'))
    plt.show()

    print("\nEnd\n")


########################################################################################################################

def zero_pad(X, pad):  # correct dims

    # X --  (m, n_H, n_W, n_C)
    # X_pad -- padded image of shape (m, n_H + 2*pad, n_W + 2*pad, n_C)

    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=(0, 0))

    return X_pad


########################################################################################################################


def conv_single_step(a_slice_prev, W, b):
    # a_slice_prev --(f, f, n_C_prev)
    # W -shape (f, f, n_C_prev)
    # b -- Bias shape (1, 1, 1)
    s = np.multiply(a_slice_prev, W)
    Z = np.sum(s)
    Z = Z + b

    return Z


########################################################################################################################

def fully_connected():
    return 0


########################################################################################################################


def conv_forward(A_prev, W, b, hparameters):  # correct dims
    """
    A_prev - shape (m, n_H_prev, n_W_prev, n_C_prev)
    W - shape (f, f, n_C_prev, n_C)
    b - (1, 1, 1, n_C)
    Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
    """

    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    (f, f, n_C_prev, n_C) = W.shape

    stride = hparameters["stride"]
    pad = hparameters["pad"]

    n_H = int((n_H_prev - f + 2 * pad) / stride) + 1
    n_W = int((n_W_prev - f + 2 * pad) / stride) + 1

    Z = np.zeros((m, n_H, n_W, n_C))

    A_prev_pad = zero_pad(A_prev, pad)

    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    # Find the corners
                    vert_start = h * stride
                    vert_end = h * stride + f
                    horiz_start = w * stride
                    horiz_end = w * stride + f

                    a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                    # print("a_slice_prev shape,W,b is:",a_slice_prev.shape ,W[:,:,:,c].shape,b[:,:,:,c].shape)

                    Z[i, h, w, c] = conv_single_step(a_slice_prev, W[:, :, :, c], b[:, :, :, c])
                    # print("Z=",Z)

    cache = (A_prev, W, b, hparameters)

    return Z, cache


########################################################################################################################


def pool_forward(A_prev, hparameters, mode="max"):  # correct dims
    """
    A_prev - shape (m, n_H_prev, n_W_prev, n_C_prev)
    A -- shape(m, n_H, n_W, n_C)
    """

    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    f = hparameters["f"]
    stride = 2  # hparameters["stride"]                                            HARDCODE

    # dimensions of the output
    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev

    A = np.zeros((m, n_H, n_W, n_C))

    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):

                    vert_start = h * stride
                    vert_end = h * stride + f
                    horiz_start = w * stride
                    horiz_end = w * stride + f

                    a_prev_slice = A_prev[i][vert_start:vert_end, horiz_start:horiz_end, c]

                    if mode == "max":
                        A[i, h, w, c] = np.max(a_prev_slice)
                    elif mode == "average":
                        A[i, h, w, c] = np.mean(a_prev_slice)

    cache = (A_prev, hparameters)
    return A, cache


########################################################################################################################

def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    """
# X  shape (input size, number of ex)
    Y shape (1, number of ex)
    """
    np.random.seed(seed)
    m = X.shape[
        0]  # training examples                                                                           XXXXXXXXXXXXXXXXX

    # Shuffle
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :]  # XXXXXXXXXXXXXXXX
    shuffled_Y = Y[permutation, :].reshape((m, 1))  # XXXXXXXXXXXXXXXX
    shuffled_Y2 = Y[permutation, :].reshape((1, m))

    print("shuffled y 2 is", shuffled_Y2)

    print("shuffled_X shape is")
    print(shuffled_X.shape)
    print("shuffled_Y shape is")
    print(shuffled_Y.shape)

    num_complete_minibatches = math.floor(m / mini_batch_size)  # number of mini batches

    mini_batches_X = np.zeros(shape=(0, 64, 784))       #CH
    mini_batches_Y = np.zeros(shape=(0, 64, 1))         #CH
    mini_batches_Y2 = np.zeros(shape=(0, 1, 64))        #CH

    for k in range(0, num_complete_minibatches):
        mini_batches_X = np.append(mini_batches_X,
                                   shuffled_X[np.newaxis, k * mini_batch_size: (k + 1) * mini_batch_size, :],
                                   axis=0)  # XXXXXXXXXXXXXXXX
        mini_batches_Y = np.append(mini_batches_Y,
                                   shuffled_Y[np.newaxis, k * mini_batch_size: (k + 1) * mini_batch_size, :],
                                   axis=0)  # XXXXXXXXXXXXXXXX
        mini_batches_Y2 = np.append(mini_batches_Y2,
                                    shuffled_Y2[:, np.newaxis, k * mini_batch_size: (k + 1) * mini_batch_size], axis=0)
    print("mini_batch_y 2 is", mini_batches_Y2[1])
    print(mini_batches_Y2.shape)

    """"
    if m % mini_batch_size != 0:
        ### START CODE HERE ### (approx. 2 lines)
        mini_batch_X = shuffled_X[np.newaxis, num_complete_minibatches * mini_batch_size:, :]                                          # XXXXXXXXXxX
        mini_batch_Y = shuffled_Y[np.newaxis, num_complete_minibatches * mini_batch_size:,:]                                           # XXXXXXXXXXXXXXX
        ### END CODE HERE ###

        mini_batches_X = np.append(mini_batches_X, mini_batch_X, axis=0)      # (939,64,784)                                 NOT ABLE TO DO LAST MINIBATCH

        mini_batches_Y = np.append(mini_batches_Y, mini_batch_Y, axis=0)
    """

    return mini_batches_X, mini_batches_Y, mini_batches_Y2


#########################################################################################################################

def initialize_adam(parameters):
    # Initializev and s as two python dictionary - keys: "dW1", "db1", ..., "dWL", "dbL"

    L = len(parameters) // 2  # number of layers in the neural networks   "//" => is used to divide with integral result
    v = {}
    s = {}

    # Initialize v, s
    for l in range(L):
        v["dW" + str(l + 1)] = np.zeros((parameters["W" + str(l + 1)].shape))
        v["db" + str(l + 1)] = np.zeros((parameters["b" + str(l + 1)].shape))
        s["dW" + str(l + 1)] = np.zeros((parameters["W" + str(l + 1)].shape))
        s["db" + str(l + 1)] = np.zeros((parameters["b" + str(l + 1)].shape))

    return v, s


def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate=0.01, beta1=0.9, beta2=0.97, epsilon=1e-8):
    L = len(parameters) // 2  # number of layers in the NN
    v_corrected = {}
    s_corrected = {}

    for l in range(L):
        #  get v_corrected
        v["dW" + str(l + 1)] = beta1 * v["dW" + str(l + 1)] + (1 - beta1) * grads['dW' + str(l + 1)]
        v["db" + str(l + 1)] = beta1 * v["db" + str(l + 1)] + (1 - beta1) * grads['db' + str(l + 1)]

        v_corrected["dW" + str(l + 1)] = v["dW" + str(l + 1)] / (1 - beta1 ** (t + 1))
        v_corrected["db" + str(l + 1)] = v["db" + str(l + 1)] / (1 - beta1 ** (t + 1))

        #  average of the squared gradients
        s["dW" + str(l + 1)] = beta2 * s["dW" + str(l + 1)] + (1 - beta2) * np.square(grads['dW' + str(l + 1)])
        s["db" + str(l + 1)] = beta2 * s["db" + str(l + 1)] + (1 - beta2) * np.square(grads['db' + str(l + 1)])

        # get bias

        s_corrected["dW" + str(l + 1)] = s["dW" + str(l + 1)] / (1 - beta2 ** (t + 1))
        s_corrected["db" + str(l + 1)] = s["db" + str(l + 1)] / (1 - beta2 ** (t + 1))

        # Update params
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * v_corrected["dW" + str(l + 1)] / (
                    np.power(s_corrected["dW" + str(l + 1)], 0.5) + epsilon)
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * v_corrected["db" + str(l + 1)] / (
                    np.power(s_corrected["db" + str(l + 1)], 0.5) + epsilon)

    return parameters, v, s


##########################################################################################################################################################

def conv_mini_to_5_dim(mini_batches_X):
    X_train_loop = []  # np.zeros(shape=(64,28,28,1))                                        #dim = (num_complete_minibatches,X_train_img.shape[0], 28, 28, 1)

    (x, y, z) = mini_batches_X.shape
    print("mini_batches_X shape in conv 5")
    print(mini_batches_X[np.newaxis, 0, 0, :, np.newaxis].reshape(28, 28).shape)

    for a in range(0, x):
        X_train_img = []
        for b in range(0, y):
            X_train_img.append(mini_batches_X[np.newaxis, a, b, :].reshape(28, 28))
        X_train_img = np.array(X_train_img)

        print("X_train_img shape is:")
        print(X_train_img.shape)  # (60000, 28, 28)
        X_train_loop.append(X_train_img)

    X_train_loop = np.array(X_train_loop)
    print("shape 0f X_train_loop is :")
    print(X_train_loop.shape)

    X_train_final = X_train_loop.reshape(X_train_loop.shape[0], 64, 28, 28, 1) / 255.0

    print("shape 0f X_train_final is :")
    print(X_train_final.shape)

    return X_train_final


########################################################################################################################

def categoricalCrossEntropy(probs, label):  # USED FOR MULTIPLE LABEL PREDICTION
    return -np.sum(label * safe_ln(probs))


########################################################################################################################

# BACKPROPOGATION

def conv_backward(dZ, cache):
    """

    dZ = gradient of the cost wrtthe output of the conv layer (Z), numpy array of shape(m, n_H, n_W, n_C)
    dA_prev = gradient of the cost wrt the input of the conv layer (A_prev), shape (m, n_H_prev, n_W_prev, n_C_prev)
    dW -- (f, f, n_C_prev, n_C)
    db --  shape (1, 1, 1, n_C)
    """

    (A_prev, W, b, hparameters) = cache

    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    (f, f, n_C_prev, n_C) = W.shape

    stride = hparameters["stride"]
    pad = hparameters["pad"]

    (m, n_H, n_W, n_C) = dZ.shape
    # initialize params
    dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))
    dW = np.zeros((f, f, n_C_prev, n_C))
    db = np.zeros((1, 1, 1, n_C))

    # Pad A_prev and dA_prev
    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)

    for i in range(m):  # loop over the training examples

        # select ith example
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]

        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h * stride
                    vert_end = h * stride + f
                    horiz_start = w * stride
                    horiz_end = w * stride + f

                    a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
                    db[:, :, :, c] += dZ[i, h, w, c]

        dA_prev[i, :, :, :] = da_prev_pad[:, :, :]

    return dA_prev, dW, db


########################################################################################################################

def create_mask_from_window(x):
    mask = (x == np.max(x))

    return mask


########################################################################################################################

def pool_backward(dA, cache, mode="max"):
    """
 backward pass of the poollayer

    Args:
    dA -- grad of cost wrt the o/p of the pooling layer, same shape = A

    Return:
    dA_prev -- gradient of cost wrt i/p of pooling layersame shape as A_prev
    """

    (A_prev, hparameters) = cache

    stride = hparameters["stride"]
    f = hparameters["f"]

    m, n_H, n_W, n_C = dA.shape

    dA_prev = np.zeros(A_prev.shape)

    for i in range(0,m):

        a_prev = A_prev[i]

        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):

                    vert_start = h * stride
                    vert_end = h * stride + f
                    horiz_start = w * stride
                    horiz_end = w * stride + f

                    if mode == "max":
                        a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]

                        mask = create_mask_from_window(a_prev_slice)
                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += mask * dA[i, h, w, c]
    return dA_prev


########################################################################################################################

def initialize_parameters_he(layers_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims) - 1  # integer representing the number of layers

    """""
    for l in range(1, L + 1):
        ### START CODE HERE ### (≈ 2 lines of code)
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * (np.sqrt(2. / layers_dims[l - 1]))          HARDCODING  FILTERS
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
        ### END CODE HERE ###

    """

    W = np.random.randn(layers_dims[0], layers_dims[1], layers_dims[2], layers_dims[3]) * (np.sqrt(2. / 2))
    b = np.zeros((1, 1, 1, layers_dims[3]))

    return W, b


########################################################################################################################

def train(X_train_final, mini_batches_Y, num_classes=10, img_depth=1, f=5, num_filt1=8, num_filt2=8,
          # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
          batch_size=64, num_epochs=2, stride=1, pad=0, save_path='params.pkl'):
    W, b = initialize_parameters_he([f, f, img_depth, num_filt1])  # f1 and f2 in parameters dictionary
    W2, b2 = initialize_parameters_he([f, f, num_filt1, num_filt2])

    w3, w4 = (128, 512), (10, 128)
    w3 = initializeWeight(w3)
    w4 = initializeWeight(w4)
    b3 = np.zeros((w3.shape[0], 1))
    b4 = np.zeros((w4.shape[0], 1))

    hparameters = {"stride": stride, "pad": pad, "f": 5}
    parameters = {"W1": W, "W2": W2, "W3": w3, "W4": w4, "b1": b, "b2": b2, "b3": b3, "b4": b4}
    v, s = initialize_adam(parameters)

    print("X_train_final size in epoch loop", X_train_final[:, :, :, :, :].shape)

    cost_record = []
    cost = 0.0
    for epoch in range(num_epochs):

        # batches = [X_train_final[k:k + batch_size] for k in range(0, X_train_final.shape[0], batch_size)]

        for batch in range(0, 3):
            # FORWARD prop
            print("batch is==", batch)
            Z, conv_forward_cache = conv_forward(X_train_final[batch, :, :, :, :], parameters["W1"], parameters["b1"],
                                                 hparameters)
            Z[Z <= 0] = 0  # pass through ReLU non-linearity
            # print("Z size in batch loop", Z.shape)

            Z2, conv_forward_cache2 = conv_forward(Z, parameters["W2"], parameters["b2"], hparameters)
            Z2[Z2 <= 0] = 0
            # print("Z2 size in batch loop", Z2.shape)

            P3, pool_cache = pool_forward(Z2, hparameters, mode="max")
            # print("P3 size in batch loop", P3.shape)
            (m, height, width, nf2) = P3.shape

            """""
            for i in range(0,64):                          # 0 is also there

                label = np.eye(num_classes)[int(mini_batches_Y[batch,:,:])].reshape(num_classes, 1)  # convert label to one-hot
                print("label size in batch loop", label)
                label_total=np.append(label_total,np.eye(num_classes)[int(mini_batches_Y[batch,i,:])].reshape(num_classes, 1),axis=0)
                #print("label_total size in batch loop", label_total.shape)
                print("i ===",i)

            """""

            print("mini batch y 2 :", mini_batches_Y[0], mini_batches_Y[batch].shape)

            # c=np.int(mini_batches_Y[batch,;,:])
            # c = list(mini_batches_Y[batch].astype(int))
            # print("c==", c[0])
            # c = np.int(c)

            a = mini_batches_Y[batch].astype(int).tolist()  # np.array([1, 0, 3])

            # a = list(repr(mini_batches_Y[batch].astype(int)))#np.array([1, 0, 3])

            # print("a =", a[0])
            a2 = np.array(a[0])
            b = np.zeros((batch_size, num_classes))  # np.zeros((3, 4))
            b[np.arange(batch_size), a2] = 1
            print("b shape:", b.shape)
            #b = b.reshape(num_classes * batch_size, 1)  # (640, 1)

            # label = np.eye(num_classes)[int(mini_batches_Y[batch,:,:])].reshape(num_classes,1)  # convert label to one-hot

            # label_total=np.array(label_total)
            print("label_total size in batch loop", b.shape)
            """""
            fc = P3[:, :, :, :].reshape((m * nf2 * height * width, 1))  # flatten pooled layer

            z = parameters["W3"].dot(fc) + parameters["b3"]  # first dense layer
            z[z <= 0] = 0  # pass through ReLU non-linearity

            out = parameters["W4"].dot(z) + parameters["b4"]  # second dense layer
            # print("out shape:", out.shape)
"""
            probs_all,dw3,dw4,db3,db4,dpool = softmaxall(P3,parameters,b)
            # probs_all= np.append(probs_all,softmax(out),axis=0)
            # probs = softmax(out)  # predict class probabilities with the softmax activation function

            # LOSS

            cost = cost + categoricalCrossEntropy(probs_all, b)  # categorical cross-entropy loss
            cost_record.append([epoch, batch, cost])
            print("cost==", cost/(batch+1))

            """""
            # BACKWARD prop
            dout = probs_all - b  # derivative of loss w.r.t. final dense layer output
            # print("dout shape is:", dout.shape)
            dw4 = dout.dot(z.T)  # loss gradient of final dense layer weights
            # print("dw4 shape is:", dw4.shape)
            db4 = np.sum(dout, axis=1).reshape(b4.shape)  # loss gradient of final dense layer biases
            # print("db4 shape is:", db4.shape)

            dz = w4.T.dot(dout)  # loss gradient of first dense layer outputs
            dz[z <= 0] = 0  # backpropagate through ReLU
            # print("dz shape is:", dz.shape)
            dw3 = dz.dot(fc.T)
            # print("dw3 shape is:", dw3.shape)
            db3 = np.sum(dz, axis=1).reshape(b3.shape)
            # print("dw3 new shape is:", dw3.shape)

            dfc = w3.T.dot(dz)  # loss gradients of fully-connected layer (pooling layer)
            # print("dfc new shape is:", dfc.shape)
            dpool = dfc.reshape(P3.shape)  # reshape fully connected into dimensions of pooling layer
"""
            print("dpool new shape is:", dpool.shape)

            dconv_forward2 = pool_backward(dpool, pool_cache, mode="max")
            # print("dconv_forward2 new shape is:", dconv_forward2.shape)
            dconv_forward2[Z2 <= 0] = 0  # backpropagate through ReLU

            dconv_forward1, dW2, db2 = conv_backward(dconv_forward2, conv_forward_cache2)
            dconv_forward1[Z <= 0] = 0  # backpropagate through ReLU
            # print("dconv_forward1 new shape is:", dconv_forward1.shape)

            dimage, dW1, db1 = conv_backward(dconv_forward1, conv_forward_cache)

            grads = {"dW1": dW1, "dW2": dW2, "dW3": dw3, "dW4": dw4, "db1": db1, "db2": db2, "db3": db3, "db4": db4}

            parameters, v, s = update_parameters_with_adam(parameters, grads, v, s, batch, learning_rate=0.01,
                                                           beta1=0.9, beta2=0.999,
                                                           epsilon=1e-8)

    cost_record = np.array(cost_record)

    return cost_record


##########################################################################################################################################################


if __name__ == "__main__":
    print("WHAT is shape of np_X_train?")
    print(np_X_train.shape)  # (60000, 784)
    print("WHAT is shape of np_Y_train?")
    print(np_y_train.shape)  # (60000, 1)

    #########=>  apply random minibatches
    mini_batches_X, mini_batches_Y, mini_batches_Y2 = random_mini_batches(np_X_test, np_y_test, mini_batch_size=64,
                                                                          seed=0)  # using test DATA !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    print("WHAT is shape of mini_batches?")

    print("WHAT is shape of mini_batch_y?")

    print("WHAT is shape of mini_batch_X?")

    ##########=> get dim (157,64,28,28,1) for test
    X_train_final = conv_mini_to_5_dim(mini_batches_X)  # ????????????????????????

    ##############################################################
    # enter n value to see nth image                             #
    # n = 0                                                       #
    # display(n)                                                  #
    #                                                            #
    ##############################################################

    cost_record = train(X_train_final, mini_batches_Y2)

    plot1(cost_record)









