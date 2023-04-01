from builtins import range
import numpy as np
from numpy.ma import maximum_fill_value


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    # print(x.shape)
    # print(w.shape)
    # print(b.shape)

    temp = x.reshape(x.shape[0], -1)
    out = temp.dot(w) + b
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    # xW+b
    
    dx = dout.dot(w.T).reshape(x.shape)
    dw = x.reshape(x.shape[0], -1).T.dot(dout)
    db = dout.sum(axis=0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    # out = x
    # out[x < 0] = 0
    # print(out)

    # out2 = np.maximum(0, x)
    # print((out == out2).all())

    out = np.maximum(0, x)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    # dx = x
    # dx[x > 0] = 1
    # dx *= dout

    dx = dout * (x > 0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #######################################################################
        # update mean
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)

        running_mean = momentum*running_mean + (1-momentum)*batch_mean
        running_var = momentum*running_var + (1-momentum)*batch_var

        x_minus_mean = x - batch_mean
        var_eps = batch_var + eps
        sqrt_var_eps = np.sqrt(var_eps)
        one_over_sqrt_var_eps = 1/np.sqrt(var_eps)

        x_hat = x_minus_mean * one_over_sqrt_var_eps
        gamma_xhat = gamma*x_hat
        out = gamma_xhat + beta
        
        cache = {
          "x": x,
          "mean": batch_mean,
          "var": batch_var,
          "eps": eps,
          "x-mean": x_minus_mean,
          "var+eps": var_eps,
          "sqrt(var+eps)": sqrt_var_eps,
          "1/sqrt(var+eps)": one_over_sqrt_var_eps,
          "x_hat": x_hat,
          "gamma": gamma,
          "gamma*x_hat": gamma_xhat,
          "beta": beta,
          "out": out
        }
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        x_hat = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma*x_hat + beta
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.
        cache = {
          "x": x,
          "mean": batch_mean,
          "var": batch_var,
          "eps": eps,
          "x-mean": x_minus_mean,
          "var+eps": var_eps,
          "sqrt(var+eps)": sqrt_var_eps,
          "1/sqrt(var+eps)": one_over_sqrt_var_eps,
          "x_hat": x_hat,
          "gamma": gamma,
          "gamma*x_hat": gamma_xhat,
          "beta": beta,
          "out": out
        }
    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    ########################################################################### 
    def plus_gate(a, b, upstream):  # a+b
      return upstream, upstream

    def minus_gate(a, b, upstream):  # a-b
      return upstream, -upstream

    def inverse_gate(a, upstream):  # 1/a
      return upstream * -a**(-2)

    def sqrt_gate(a, upstream):   # sqrt(a)
      return upstream * 1/2*a**(-1/2)

    def element_wise_multiply_gate(a, b, upstream): # a*b
      return upstream * b, upstream * a

    def dot_gate(a, b, upstream): # a dot b
      # input: a: (C, D)   b:(D, E)  upstream:(C, E)
      # return: da, db
      return upstream.dot(b.T), a.T.dot(upstream)

    
    x = cache["x"]
    mean = cache["mean"] 
    var = cache["var"]
    eps = cache["eps"]
    x_minus_mean = cache["x-mean"]
    var_eps = cache["var+eps"]
    sqrt_var_eps = cache["sqrt(var+eps)"]
    one_over_sqrt_var_eps = cache["1/sqrt(var+eps)"]
    x_hat = cache["x_hat"]  
    gamma = cache["gamma"]  
    gamma_xhat = cache["gamma*x_hat"] 
    beta = cache["beta"]  
    out = cache["out"]  
    N = x.shape[0]

    dy = dout
    dgamma_xhat, dbeta = plus_gate(None, None, dy)
    dbeta = np.sum(dbeta, axis=0)

    dx_hat, dgamma = element_wise_multiply_gate(x_hat, gamma, dgamma_xhat)
    dgamma = np.sum(dgamma, axis=0)

    dx_minus_mean, ds4 = \
      element_wise_multiply_gate(x_minus_mean, one_over_sqrt_var_eps, dx_hat)

    done_over_sqrt_var_eps = inverse_gate(sqrt_var_eps, ds4)
    dsqrt_var_eps = sqrt_gate(var_eps, done_over_sqrt_var_eps)
    dvar, deps = plus_gate(var, eps, dsqrt_var_eps)
    dvar = np.sum(dvar, axis=0)
    dpes = np.sum(deps)

    dx1 = dmean = dx2 = 0
    dx1, dmean = minus_gate(x, mean, dx_minus_mean)
    dmean = np.sum(dmean, axis=0)
    dmean += (dvar * 2 * (x-mean) * -1 / N).sum(axis=0)
    dx2 = (dvar * 2 * (x-mean)) / N
    
    dx = dx1 + dx2 + dmean * 1 / N

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # dL/dxhat = dL/dy * gamma
    mu = cache["mean"]
    gamma = cache["gamma"]
    dxhat = dout * gamma

    dgamma = (dout * cache["x_hat"]).sum(axis=0)
    dbeta = dout.sum(axis=0)
    # dxhat/dx = dxhat/dx + dxhat/dmu * dmu/dx + dxhat/dvar * (dvar/dx + dvar/dmu*dmu/dx)
    x = cache["x"]
    N = x.shape[0]
    var = cache["var"]
    eps = cache["eps"]
    x_hat = cache["x_hat"]

    # dxhat_dx = np.ones(x.shape) / np.sqrt(var+eps)

    # dxhat_dmu = -1/np.sqrt(var+eps)
    # dmu_dx = 1/N * np.ones(x.shape)

    # dxhat_dvar = -1/2 * (var+eps)**(-3/2) * (x-mu)
    # dvar_dx = 2*(x-mu)/N
    # dvar_dmu = np.sum(-2*(x-mu)/N, axis=0)

    # dxhat_dx = dxhat_dx + dxhat_dmu*dmu_dx + dxhat_dvar*(dvar_dx + dvar_dmu*dmu_dx)

    # dL/dx = dL/dxhat * dxhat/dx
    # dx = dxhat * dxhat_dx
    # print(dx.sum())

    # a = dout* (1/np.sqrt(var+eps) -  1 / (N * np.sqrt(var+eps)) * \
      # (-1/2)*(x-mu)*(var+eps)**(-3/2)*(2/N)*(x-mu) *  \
      # (-1/2)*(x-mu)*(var+eps)**(-3/2)*(-2/N)*(x-mu)*(1/N))
    # print(a.sum())
    
    # dx = dxhat*a
    dx = batchnorm_backward(dout, cache)[0]

    # print(dx.sum())

    # return batchnorm_backward(dout, cache)
    ##
    # N, D = dout.shape
    # x, est_x, gamma, beta, mean, var, eps= cache
    # dgamma = np.sum(dout * est_x, axis = 0) 
    # dbeta = np.sum(dout, axis = 0)
    # dest_x = dout * gamma
    # destivar = dest_x * np.sqrt(var+eps)
    # dx = (-np.mean(destivar, axis=0) - np.mean(dest_x*est_x, axis=0) * (x-mean) + destivar) / (var+eps)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # p = dropout_param["p"]
        # seed = dropout_param["seed"]
        mask = np.random.rand(x.size).reshape(x.shape)
        mask = mask < (1-p)
        mask = mask / (1-p)

        out = mask * x
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        out = x
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        dx = dout * mask
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    stride = conv_param["stride"]
    pad = conv_param["pad"]

    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    
    x_pad = np.pad(x, [(0,), (0,), (pad,), (pad,)], 'constant')

    out_H = 1 + ((H + 2 * pad - HH) // stride)
    out_W = 1 + ((W + 2 * pad - WW) // stride)

    out = np.zeros((N, F, out_H, out_W))
    
    for row in range(out_H):
      for col in range(out_W):
        region = x_pad[:, :, row*stride: row*stride+HH, col*stride: col*stride+WW]

        for f in range(F):
          out[:, f, row, col] = np.sum(region * w[f, :, :, :], axis=(1, 2, 3))
          out[:, f, row, col] += b[f]
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives. - (N, F, out_H, out_W)
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    x, w, b, conv_param = cache

    N, C, H, W = x.shape
    F, C, HH, WW = w.shape

    stride, pad = conv_param["stride"], conv_param["pad"]

    x_pad = np.pad(x, [(0,), (0,), (pad,), (pad,)], 'constant')
    out_H = 1 + ((H + 2 * pad - HH) // stride)
    out_W = 1 + ((W + 2 * pad - WW) // stride)

    dx = np.zeros(x.shape)
    dx_pad = np.zeros(x_pad.shape)
    dw = np.zeros(w.shape)
    db = np.zeros(b.shape)
    
    for row in range(out_H):
      for col in range(out_W):
        region = x_pad[:, :, row*stride: row*stride+HH, col*stride: col*stride+WW]
        
        for f in range(F):
          dneurons = dout[:, f, row, col].reshape(-1, 1, 1, 1)  # accross channel N
          dw[f, :, :, :] += np.sum(dneurons * region, axis=0)
          # for n in range(N):
            # dw[f, :, :, :] += np.sum(dneurons[n, :, :, :] * region[n, :, :, :])
        for n in range(N):
          dneurons = dout[n, :, row, col].reshape(-1, 1, 1, 1)  # accross channel F
          dx_pad[n, :, row*stride: row*stride+HH, col*stride: col*stride+WW] += \
            np.sum(w * dneurons, axis=0)

    dx = dx_pad[:, :, pad:-pad, pad:-pad]
    db = np.sum(dout, axis=(0, 2, 3))

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max pooling forward pass                            #
    ###########################################################################
    N, C, H, W = x.shape
    stride = pool_param["stride"]
    pool_height = pool_param["pool_height"]
    pool_width = pool_param["pool_width"]

    out_H = 1 + (H - pool_height) // stride
    out_W = 1 + (H - pool_width) // stride

    out = np.zeros((N, C, out_H, out_W))
    for row in range(out_H):
      for col in range(out_W):
        region = x[:, :, row*stride: row*stride+pool_height, col*stride: col*stride+pool_height]
        out[:, :, row, col] = np.max(region, axis=(2, 3))

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max pooling backward pass                           #
    ###########################################################################
    x, pool_param = cache
    N, C, H, W = x.shape
    stride = pool_param["stride"]
    pool_height = pool_param["pool_height"]
    pool_width = pool_param["pool_width"]

    out_H = 1 + (H - pool_height) // stride
    out_W = 1 + (H - pool_width) // stride

    dx = np.zeros(x.shape)
    # dout shape: (N, C, out_H, out_W)
    for row in range(out_H):
      for col in range(out_W):
        region = x[:, :, row*stride: row*stride+pool_height, col*stride: col*stride+pool_height]

        maximum_fill_value = np.max(region, axis=(2, 3), keepdims=True)
        taken_positions = (region == maximum_fill_value)

        dx[:, :, row*stride: row*stride+pool_height, col*stride: col*stride+pool_height] += taken_positions*dout[:, :, row, col].reshape(N, C, 1, 1)

    # for n in range(N):
    #   for c in range(C):
    #     for h in range(out_H):
    #       for w in range(out_W):
    #         region = x[n, c, h*stride: h*stride+pool_height, w: w*stride+pool_width]

    #         maximum_fill_index = np.argmax(region, axis=0)
            
    #         for i in range(h, h+pool_height):
    #           dx[n, c, i, maximum_fill_index] *= dout[n, c, h, w]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    N, C, H, W = x.shape

    x = x.transpose(0, 2, 3, 1)
    x = x.reshape(N*H*W, C)

    out, cache = batchnorm_forward(x, gamma, beta, bn_param)

    out = out.reshape(N, H, W, C)
    out = out.transpose(0, 3, 1, 2)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    N, C, H, W = dout.shape

    dout = dout.transpose(0, 2, 3, 1)
    dout = dout.reshape(N*H*W, C)

    dx, dgamma, dbeta = batchnorm_backward(dout, cache)

    dx = dx.reshape(N, H, W, C)
    dx = dx.transpose(0, 3, 1, 2)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
