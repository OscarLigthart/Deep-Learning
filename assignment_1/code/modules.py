"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np

class LinearModule(object):
  """
  Linear module. Applies a linear transformation to the input data. 
  """
  def __init__(self, in_features, out_features):
    """
    Initializes the parameters of the module. 
    
    Args:
      in_features: size of each input sample
      out_features: size of each output sample

    TODO:
    Initialize weights self.params['weight'] using normal distribution with mean = 0 and 
    std = 0.0001. Initialize biases self.params['bias'] with 0. 
    
    Also, initialize gradients with zeros.
    """
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################

    # mean and std for initializing weight matrix
    mean = 0
    std_dev = 0.0001

    # create weight matrix
    weight = np.random.normal(mean, std_dev, (out_features, in_features))

    # initialize gradient matrix for weights
    grad_weight = np.zeros((in_features, out_features))

    # create biases (in batches)
    bias = np.zeros(out_features)
    grad_bias = np.zeros(out_features)

    self.params = {'weight': weight, 'bias': bias}
    self.grads = {'weight': bias, 'bias': grad_bias}

    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################

    # expand batch dimension
    self.input = x
    out = np.matmul(x, self.params['weight'].T) + np.expand_dims(self.params['bias'],0)

    self.value = out

    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous module
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module. Store gradient of the loss with respect to 
    layer parameters in self.grads['weight'] and self.grads['bias']. 
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################

    # get dx
    dx = np.matmul(dout, self.params['weight'])

    # get db
    self.grads['bias'] = np.mean(np.matmul(dout, np.identity(len(self.params['bias']))),0)

    # get dW
    self.grads['weight'] = np.matmul(dout.T, self.input)

    ########################
    # END OF YOUR CODE    #
    #######################
    
    return dx

class ReLUModule(object):
  """
  ReLU activation module.
  """
  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################

    out = np.maximum(np.zeros(x.shape), x)
    self.value = out

    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous module
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################

    # 1 if > 0 and 0 if < 0
    ddout = np.zeros_like(self.value)
    ddout[self.value > 0] = 1

    # multiply with previous gradient
    dx = dout * ddout

    ########################
    # END OF YOUR CODE    #
    #######################    

    return dx

class SoftMaxModule(object):
  """
  Softmax activation module.
  """
  def forward(self, x):
    """
    Forward pass.
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################

    # get maximum of each row
    b = x.max(1)
    b = np.expand_dims(b,1)
    y = np.exp(x - b)

    # apply softmax to each row
    out = (y.T / y.sum(1)).T
    self.value = out

    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous module
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################

    # get output of previous model, since softmax has already been applied there
    soft = self.value

    # create first matrix consisting of i != j derivatives
    ss1 = np.einsum('ij,ik->ijk', -soft, soft)

    # create second matrix consisting of i == j derivatives
    ss2 = np.einsum('ij,ik->ijk', soft, (np.ones_like(soft) - soft))

    # overwrite diagonal of first matrix with diagonal of second matrix
    dimsize = np.size(ss1, 1)
    ss1[:, np.diag_indices(dimsize)[0], np.diag_indices(dimsize)[1]] = ss2[:, np.diag_indices(dimsize)[0], np.diag_indices(dimsize)[1]]

    # multiply out gradient with derivatives
    dx = np.einsum('ij, ijk -> ik',dout, ss1)

    ########################
    # END OF YOUR CODE    #
    #######################

    return dx

class CrossEntropyModule(object):
  """
  Cross entropy loss module.
  """
  def forward(self, x, y):
    """
    Forward pass.

    Args:
      x: input to the module
      y: labels of the input
    Returns:
      out: cross entropy loss
    
    TODO:
    Implement forward pass of the module. 
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    out = -np.log(x[np.arange(x.shape[0]), y.argmax(1)]).mean()

    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, x, y):
    """
    Backward pass.

    Args:
      x: input to the module
      y: labels of the input
    Returns:
      dx: gradient of the loss with the respect to the input x.
    
    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################

    dx = -(y/x)/y.shape[0]

    ########################
    # END OF YOUR CODE    #
    #######################

    return dx

