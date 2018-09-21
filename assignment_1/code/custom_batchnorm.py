import numpy as np
import torch
import torch.nn as nn

"""
The modules/function here implement custom versions of batch normalization in PyTorch.
In contrast to more advanced implementations no use of a running mean/variance is made.
You should fill in code into indicated sections.
"""

######################################################################################
# Code for Question 3.1
######################################################################################

class CustomBatchNormAutograd(nn.Module):
  """
  This nn.module implements a custom version of the batch norm operation for MLPs.
  The operations called in self.forward track the history if the input tensors have the
  flag requires_grad set to True. The backward pass does not need to be implemented, it
  is dealt with by the automatic differentiation provided by PyTorch.
  """

  def __init__(self, n_neurons, eps=1e-5):
    """
    Initializes CustomBatchNormAutograd object. 
    
    Args:
      n_neurons: int specifying the number of neurons
      eps: small float to be added to the variance for stability
    
    TODO:
      Save parameters for the number of neurons and eps.
      Initialize parameters gamma and beta via nn.Parameter
    """
    super(CustomBatchNormAutograd, self).__init__()

    ########################
    # PUT YOUR CODE HERE  #
    #######################

    self.neurons = n_neurons
    self.epsilon = eps

    # initialize gamma and beta
    self.gamma = nn.Parameter(torch.Tensor(np.ones(n_neurons)))
    self.beta = nn.Parameter(torch.Tensor(np.zeros(n_neurons)))

    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, input):
    """
    Compute the batch normalization
    
    Args:
      input: input tensor of shape (n_batch, n_neurons)
    Returns:
      out: batch-normalized tensor
    
    TODO:
      Check for the correctness of the shape of the input tensor.
      Implement batch normalization forward pass as given in the assignment.
      For the case that you make use of torch.var be aware that the flag unbiased=False should be set.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    assert(input.shape[1] == self.neurons)

    input_mean = input.mean(0)

    # get variance
    var = ((input-input_mean)**2).mean(0)

    # divide normalized input by ivar
    norm = (input-input_mean)/(var + self.epsilon).sqrt()

    out = (self.gamma*norm) + self.beta

    ########################
    # END OF YOUR CODE    #
    #######################

    return out



######################################################################################
# Code for Question 3.2 b)
######################################################################################


class CustomBatchNormManualFunction(torch.autograd.Function):
  """
  This torch.autograd.Function implements a functional custom version of the batch norm operation for MLPs.
  Using torch.autograd.Function allows you to write a custom backward function.
  The function will be called from the nn.Module CustomBatchNormManualModule
  Inside forward the tensors are (automatically) not recorded for automatic differentiation since the backward
  pass is done via the backward method.
  The forward pass is not called directly but via the apply() method. This makes sure that the context objects
  are dealt with correctly. Example:
    my_bn_fct = CustomBatchNormManualFunction()
    normalized = fct.apply(input, gamma, beta, eps)
  """

  @staticmethod
  def forward(ctx, input, gamma, beta, eps=1e-5):
    """
    Compute the batch normalization
    
    Args:
      ctx: context object handling storing and retrival of tensors and constants and specifying
           whether tensors need gradients in backward pass
      input: input tensor of shape (n_batch, n_neurons)
      gamma: variance scaling tensor, applied per neuron, shpae (n_neurons)
      beta: mean bias tensor, applied per neuron, shpae (n_neurons)
      eps: small float added to the variance for stability
    Returns:
      out: batch-normalized tensor

    TODO:
      Implement the forward pass of batch normalization
      Store constant non-tensor objects via ctx.constant=myconstant
      Store tensors which you need in the backward pass via ctx.save_for_backward(tensor1, tensor2, ...)
      Intermediate results can be decided to be either recomputed in the backward pass or to be stored
      for the backward pass. Do not store tensors which are unnecessary for the backward pass to save memory!
      For the case that you make use of torch.var be aware that the flag unbiased=False should be set.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################

    N, D = input.shape

    # get mu
    mu = input - ((1/N) * input.sum(0))

    # get var and inversed var
    var = (1/N) * (mu.pow(2)).sum(0)

    ivar = 1/(var+eps).sqrt()

    # get xhat
    xhat = mu * ivar

    # apply final function
    out = (gamma * xhat) + beta

    # save tensors for gradient calculation
    ctx.save_for_backward(input, gamma, beta, mu, var, xhat, out)
    ctx.eps = eps

    ########################
    # END OF YOUR CODE    #
    #######################

    return out


  @staticmethod
  def backward(ctx, grad_output):
    """
    Compute backward pass of the batch normalization.
    
    Args:
      ctx: context object handling storing and retrival of tensors and constants and specifying
           whether tensors need gradients in backward pass
    Returns:
      out: tuple containing gradients for all input arguments
    
    TODO:
      Retrieve saved tensors and constants via ctx.saved_tensors and ctx.constant
      Compute gradients for inputs where ctx.needs_input_grad[idx] is True. Set gradients for other
      inputs to None. This should be decided dynamically.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    input, gamma, beta, mu, var, xhat, out = ctx.saved_tensors

    # get inversed variance
    ivar = 1/(var + ctx.eps).sqrt()

    # get the shape
    N,D = grad_output.shape

    # get beta and gamma gradient
    grad_beta = torch.sum(grad_output, 0)
    grad_gamma = torch.sum((xhat * grad_output), 0)

    # continue gradient of x
    grad_xhat = gamma * grad_output #grad_x

    # get the gradient wrt variance
    grad_ivar = torch.sum(grad_xhat*mu, 0) #grad_var

    # continue gradient of x
    grad_x1 = ivar * grad_xhat

    # get gradient wrt var
    grad_var = grad_ivar * -1 / (var + ctx.eps)
    grad_var = 0.5 * 1 / (var + ctx.eps).sqrt() * grad_var
    grad_var = 1 / N * grad_var
    grad_x2 = 2 * mu * grad_var
    grad_x_var = grad_x1 + grad_x2

    # get gradient wrt mu
    grad_mu = -1 * torch.sum(grad_x1, 0)

    # calculate gradient of x wrt mu
    grad_x_mu = 1/ N * grad_mu

    # get input gradient
    grad_input = grad_x_var + grad_x_mu

    ########################
    # END OF YOUR CODE    #
    #######################

    # return gradients of the three tensor inputs and None for the constant eps
    return grad_input, grad_gamma, grad_beta, None



######################################################################################
# Code for Question 3.2 c)
######################################################################################

class CustomBatchNormManualModule(nn.Module):
  """
  This nn.module implements a custom version of the batch norm operation for MLPs.
  In self.forward the functional version CustomBatchNormManualFunction.forward is called.
  The automatic differentiation of PyTorch calls the backward method of this function in the backward pass.
  """

  def __init__(self, n_neurons, eps=1e-5):
    """
    Initializes CustomBatchNormManualModule object.
    
    Args:
      n_neurons: int specifying the number of neurons
      eps: small float to be added to the variance for stability
    
    TODO:
      Save parameters for the number of neurons and eps.
      Initialize parameters gamma and beta via nn.Parameter
    """
    super(CustomBatchNormManualModule, self).__init__()

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    self.neurons = n_neurons
    self.epsilon = eps

    # initialize gamma and beta
    self.gamma = nn.Parameter(torch.Tensor(np.ones(n_neurons)))
    self.beta = nn.Parameter(torch.Tensor(np.zeros(n_neurons)))

    self.fct = CustomBatchNormManualFunction()
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, input):
    """
    Compute the batch normalization via CustomBatchNormManualFunction
    
    Args:
      input: input tensor of shape (n_batch, n_neurons)
    Returns:
      out: batch-normalized tensor
    
    TODO:
      Check for the correctness of the shape of the input tensor.
      Instantiate a CustomBatchNormManualFunction.
      Call it via its .apply() method.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################

    out = self.fct.apply(input, self.gamma, self.beta, self.epsilon)

    #######################
    # END OF YOUR CODE    #
    #######################

    return out
