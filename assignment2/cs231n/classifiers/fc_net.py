from binascii import b2a_base64
from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *
from zmq import XPUB


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases using the keys 'W2' and 'b2'.                                 #
        ############################################################################
        W1_shape = (input_dim, hidden_dim)
        b1_shape = hidden_dim,

        W1 = np.random.normal(loc=0, scale=weight_scale, size=W1_shape)
        b1 = np.zeros(b1_shape)

        W2_shape = (hidden_dim, num_classes)
        b2_shape = num_classes,

        W2 = np.random.normal(loc=0, scale=weight_scale, size=W2_shape)
        b2 = np.zeros(b2_shape)

        self.params["W1"] = W1
        self.params["b1"] = b1
        self.params["W2"] = W2
        self.params["b2"] = b2
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        W1 = self.params["W1"]
        b1 = self.params["b1"]
        W2 = self.params["W2"]
        b2 = self.params["b2"]

        X = X.reshape(X.shape[0], -1)
        X2, cache1 = affine_relu_forward(X, W1, b1)
        scores, cache2 = affine_forward(X2, W2, b2)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, softmax_grad = softmax_loss(scores, y)
        loss += self.reg / 2 * ((W1*W1).sum() + (W2*W2).sum())

        X2_grad, W2_grad, b2_grad = affine_backward(softmax_grad, cache2)
        X_grad, W1_grad, b1_grad = affine_relu_backward(X2_grad, cache1)

        grads["W1"] = W1_grad + self.reg * W1
        grads["b1"] = b1_grad
        grads["W2"] = W2_grad + self.reg * W2
        grads["b2"] = b2_grad
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to one and shift      #
        # parameters should be initialized to zero.                                #
        ############################################################################
        num_layer = 1
        W_shape = None
        b_shape = None
        for num_layer in range(1, self.num_layers):
          W_shape = (input_dim, hidden_dims[num_layer-1])
          b_shape = hidden_dims[num_layer-1],
          input_dim = hidden_dims[num_layer-1]

          W = np.random.normal(loc=0, scale=weight_scale, size=W_shape)
          b = np.zeros(b_shape)

          self.params["W"+str(num_layer)] = W
          self.params["b"+str(num_layer)] = b

          if use_batchnorm:
            self.params["gamma"+str(num_layer)] = np.ones(hidden_dims[num_layer-1])
            self.params["beta"+str(num_layer)] = np.zeros(hidden_dims[num_layer-1])

        if self.num_layers == 1:
          W_shape = (input_dim, num_classes)
          b_shape = num_classes,
        else:
          num_layer += 1
          W_shape = (hidden_dims[num_layer-2], num_classes)
          b_shape = num_classes,

        self.params["W"+str(num_layer)] = np.random.normal(loc=0, scale=weight_scale, size=W_shape)
        self.params["b"+str(num_layer)] = np.zeros(b_shape)
        # print(self.params.keys())
        # for a,b in self.params.items():
          # print(a, b.shape)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        out = {0: X}
        caches = {}

        num_layer = 1
        for num_layer in range(1, self.num_layers):
          W = self.params["W"+str(num_layer)]
          b = self.params["b"+str(num_layer)]

          if self.use_batchnorm:
            gamma = self.params["gamma"+str(num_layer)]
            beta = self.params["beta"+str(num_layer)]
            out[num_layer], caches[num_layer] = affine_batchnorm_relu_forward(out[num_layer-1], W, b, gamma, beta, self.bn_params[num_layer-1])
          else:
            out[num_layer], caches[num_layer] = affine_relu_forward(out[num_layer-1], W, b)

          if self.use_dropout:  
            out[num_layer], caches["dropout"+str(num_layer)] = dropout_forward(out[num_layer], self.dropout_param)

        if self.num_layers == 1:
          out[num_layer], caches[num_layer] = affine_forward(out[num_layer-1], W, b)
        else:
          num_layer += 1
          W = self.params["W"+str(num_layer)]
          b = self.params["b"+str(num_layer)]
          out[num_layer], caches[num_layer] = affine_forward(out[num_layer-1], W, b)

        scores = out[num_layer]
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, softmax_grad = softmax_loss(scores, y)
        for i in range(1, self.num_layers+1):
          W = self.params['W'+str(i)]
          loss += 0.5 * self.reg * (W*W).sum()
        
        num_layer = self.num_layers
        dX, dW, db = affine_backward(softmax_grad, caches[num_layer])
        grads["W"+str(num_layer)] = dW + self.reg * self.params["W"+str(num_layer)]
        grads["b"+str(num_layer)] = db

        for num_layer in range(self.num_layers-1, 0, -1):
          if self.use_dropout:
            dX = dropout_backward(dX, caches["dropout"+str(num_layer)])

          if self.use_batchnorm:
            dX, dW, db, dgamma, dbeta = \
              affine_batchnorm_relu_backward(dX, caches[num_layer])

            grads["gamma"+str(num_layer)] = dgamma
            grads["beta"+str(num_layer)] = dbeta
          else:
            # print(num_layer, dX.shape)
            dX, dW, db = affine_relu_backward(dX, caches[num_layer])

          grads["W"+str(num_layer)] = dW + self.reg * self.params["W"+str(num_layer)]
          grads["b"+str(num_layer)] = db
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
