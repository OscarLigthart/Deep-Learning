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
    mean = 0
    std_dev = 0.0001
    print(in_features)
    print(out_features)
    # create weight matrices
    weight = np.random.normal(mean, std_dev, (out_features, in_features))
    print(weight.shape)
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
    #print('dout')
    #print(dout.shape)

    #print('dx')
    #print(dx.shape) # check!

    #print('weights')
    #print(self.params['weight'].shape)

    # get db
    self.grads['bias'] = np.mean(np.matmul(dout, np.identity(len(self.params['bias']))),0)
    #print('bias')
    #print(self.grads['bias'].shape)

    # get dW
    self.grads['weight'] = np.matmul(dout.T, self.input)
    #print('gradients')
    #print(self.grads['weight'].shape)

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

    print(dout.shape)
    #d = np.ones_like(dout)
    ddout = np.zeros_like(self.value)


    ddout[self.value > 0] = 1

    dx = dout * ddout

    ########################
    # END OF YOUR CODE    #
    #######################    
    print(dx.shape)
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
    #print(out)
    #print(out.shape)

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

    #SM = self.value.reshape((-1, 1))
    #print(SM.shape)
    # softmax output: self.value
    print(self.value.shape)
    # first create -SS matrix
    #soft = np.expand_dims(self.value, 1)
    soft = self.value
    print(soft.shape)

    ss1 = np.einsum('ij,ik->ijk', -soft, soft)

    ss2 = np.einsum('ij,ik->ijk', soft, (np.ones_like(soft) - soft))

    diag_ss2 = ss2.diagonal(0,1,2)

    # overwrite diagonal
    dimsize = np.size(ss1, 1)
    ss1[:, np.diag_indices(dimsize)[0], np.diag_indices(dimsize)[1]] = ss2[:, np.diag_indices(dimsize)[0], np.diag_indices(dimsize)[1]]

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

'''
y = np.array([[0,1,0],
              [1,0,0],
              [1,0,0]])

x = np.array([[0.2,0.4,0.4],
              [0.3,0.3,0.3],
              [0.7,0.0,0.0]])

x = np.array([[0.2,0.4,-0.4],
              [0.3,-0.3,0.3],
              [-0.7,0.0,-40.0]])

a = np.array([0.2,0.4,0.4])
b = np.array([0.3,0.3,0.3])

loss = ReLUModule()
loss.forward(x)
loss.backward(x)

'''