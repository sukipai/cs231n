from builtins import range
from builtins import object
import os
import numpy as np

from ..layers import *
from ..layer_utils import *


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

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        # 初始化第一层的权值和偏置，input_dim是输入维度，hidden_dim是隐藏层维度，num_classes是输出维度
        # 输入是(1, input_dim),所以第一层的权值应该是(input_dim, hidden_dim)
        W1 = weight_scale * np.random.randn(input_dim, hidden_dim)
        b1 = np.zeros(hidden_dim)

        # 初始化第二层的权值和偏置，hidden_dim是隐藏层维度，num_classes是输出维度
        # 输入是(1, hidden_dim),所以第二层的权值应该是(hidden_dim, num_classes)
        W2 = weight_scale * np.random.randn(hidden_dim, num_classes)
        b2 = np.zeros(num_classes)

        # 将权值和偏置存储到params字典中
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
        # x_reshaped是(N, D), W1是(D, H), 
        W1 = self.params["W1"]
        b1 = self.params["b1"]
        layer1, cache1 = affine_relu_forward(X, W1, b1)

        W2 = self.params["W2"]
        b2 = self.params["b2"]
        scores, cache2 = affine_forward(layer1, W2, b2)
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
        data_loss, dScores = softmax_loss(scores, y)

        reg_loss = 0.5 * self.reg * (np.sum(W1**2) + np.sum(W2**2))
        loss = data_loss + reg_loss

        dHidden, dW2, db2 = affine_backward(dScores, cache2)
        dW2 += self.reg * W2

        dX, dW1, db1 = affine_relu_backward(dHidden, cache1)
        dW1 += self.reg * W1

        grads["W1"] = dW1
        grads["b1"] = db1
        grads["W2"] = dW2
        grads["b2"] = db2
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

    def save(self, fname):
      """Save model parameters."""
      fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
      params = self.params
      np.save(fpath, params)
      print(fname, "saved.")
    
    def load(self, fname):
      """Load model parameters."""
      fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
      if not os.path.exists(fpath):
        print(fname, "not available.")
        return False
      else:
        params = np.load(fpath, allow_pickle=True).item()
        self.params = params
        print(fname, "loaded.")
        return True



class FullyConnectedNet(object):
    """Class for a multi-layer fully connected neural network.

    Network contains an arbitrary number of hidden layers, ReLU nonlinearities,
    and a softmax loss function. This will also implement dropout and batch/layer
    normalization as options. For a network with L layers, the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional and the {...} block is
    repeated L - 1 times.

    Learnable parameters are stored in the self.params dictionary and will be learned
    using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout_keep_ratio=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout_keep_ratio: Scalar between 0 and 1 giving dropout strength.
            If dropout_keep_ratio=1 then the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
            are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
            initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
            this datatype. float32 is faster but less accurate, so you should use
            float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers.
            This will make the dropout layers deteriminstic so we can gradient check the model.
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        layers_dims = [input_dim] + hidden_dims + [num_classes]

        for i in range(1, self.num_layers + 1):
            W_name = "W" + str(i)
            b_name = "b" + str(i)

            previous_dim = layers_dims[i - 1]
            current_dim = layers_dims[i]

            Wi = weight_scale * np.random.randn(previous_dim, current_dim)
            bi = np.zeros(current_dim)

            self.params[W_name] = Wi
            self.params[b_name] = bi

            if self.normalization is not None and i < self.num_layers:
                gamma_name = "gamma" + str(i)
                beta_name = "beta" + str(i)

                self.params[gamma_name] = np.ones(current_dim)
                self.params[beta_name] = np.zeros(current_dim)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype.
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """Compute loss and gradient for the fully connected net.
        
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
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully connected net, computing  #
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
        caches = []
        current_input = X

        # 循环L - 1个隐藏层
        for i in range(1, self.num_layers):
            Wi = self.params[f"W{i}"]
            bi = self.params[f"b{i}"]

            # 1. affine层
            out_affine, fc_cache = affine_forward(current_input, Wi, bi)

            # 2.normalization层
            out_norm = out_affine
            bn_cache = None
            if self.normalization == "batchnorm":
                gamma = self.params[f"gamma{i}"]
                beta = self.params[f"beta{i}"]

                bn_param = self.bn_params[i - 1]
                out_norm, bn_cache = batchnorm_forward(out_affine, gamma, beta, bn_param)
            elif self.normalization == "layernorm":
                gamma = self.params[f"gamma{i}"]
                beta = self.params[f"beta{i}"]

                ln_param = {}
                out_norm, bn_cache = layernorm_forward(out_affine, gamma, beta, ln_param)
            
            # 3. ReLU层
            out_relu, relu_cache = relu_forward(out_norm)

            # 4. Dropout层
            out_dropout = out_relu
            dropout_cache = None
            if self.use_dropout:
                out_dropout, dropout_cache = dropout_forward(out_relu, self.dropout_param)
            
            cache = {
                "fc":       fc_cache,
                "bn":       bn_cache,
                "relu":     relu_cache,
                "dropout":  dropout_cache
            }

            caches.append(cache)

            current_input = out_dropout

        # 处理最后一层
        W_last = self.params[f"W{self.num_layers}"]
        b_last = self.params[f"b{self.num_layers}"]
        scores, last_cache = affine_forward(current_input, W_last, b_last)
        # 最后一个affine的cache要存入caches
        cache = {
            "fc":       last_cache,
            "bn":       None,
            "relu":     None,
            "dropout":  None
        }
        caches.append(cache)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early.
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the   #
        # scale and shift parameters.                                              #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # 1. 计算loss
        loss, dscore = softmax_loss(scores, y)

        # L2正则化Loss
        reg_loss = 0.0
        for i in range(1, self.num_layers + 1):
            W = self.params[f"W{i}"]
            reg_loss += 0.5 * self.reg * np.sum(W * W)
        loss += reg_loss

        # 2. 反向传播
        # 最后一层只有affine层所以只需要affine_backward
        last_cache = caches[self.num_layers - 1]["fc"]
        upstream_grad, grads[f"W{self.num_layers}"], grads[f"b{self.num_layers}"] = affine_backward(dscore, last_cache)

        grads[f"W{self.num_layers}"] += self.reg * self.params[f"W{self.num_layers}"]

        # 3. 循环向前依次反向传播
        # 对于每一层都是，affine - normalization - relu - dropout
        # 所以反响传播时候需要：依次处理dropout - relu - normalization - affine
        for i in reversed(range(1, self.num_layers)):
            cache = caches[i - 1]

            # 第一步，处理dropout的反向传播传播
            if self.use_dropout:
                upstream_grad = dropout_backward(upstream_grad, cache["dropout"])
            
            # 第二步，处理ReLU的反响传播
            upstream_grad = relu_backward(upstream_grad, cache["relu"])

            # 第三步，处理normalization的反向传播
            if self.normalization == "batchnorm":
                upstream_grad, grads[f"gamma{i}"], grads[f"beta{i}"] = batchnorm_backward(upstream_grad, cache["bn"])
            elif self.normalization == "layernorm":
                upstream_grad, grads[f"gamma{i}"], grads[f"beta{i}"] = layernorm_backward(upstream_grad, cache["bn"])
            
            # 第四步，处理affine的反向传播
            upstream_grad, grads[f"W{i}"], grads[f"b{i}"] = affine_backward(upstream_grad, cache["fc"])
            grads[f"W{i}"] += self.reg * self.params[f"W{i}"]

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


    def save(self, fname):
      """Save model parameters."""
      fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
      params = self.params
      np.save(fpath, params)
      print(fname, "saved.")
    
    def load(self, fname):
      """Load model parameters."""
      fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
      if not os.path.exists(fpath):
        print(fname, "not available.")
        return False
      else:
        params = np.load(fpath, allow_pickle=True).item()
        self.params = params
        print(fname, "loaded.")
        return True