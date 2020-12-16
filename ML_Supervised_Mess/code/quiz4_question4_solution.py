import numpy as np

def main():
  
  ## Load the data
  m0=100                             ## m0+1 gives the index of the 0 input  
  m=2*m0+1                           ## number of points
  xx=2*np.arange(m+1)/m-1            ## input data is in the range [-1,+1], with steps 0.01 
  rr=xx**2                           ## output: parabola

  ## set the learner parameters
  niteration=100                     ## number of iteations
  H=5                                ## number of nodes in the hidden layer
  eta=0.3                            ## learning speed, step size

  ## Random initialization of the edge weights
  ## Set the seed to a fixed number
  np.random.seed(12345) 
  ## weights between the input and the hidden layers
  W=2*(np.random.rand(H)-0.5)/100    ## random uniform in [-0.01,0.01]
  ## weights between the hidden and the output layers
  V=2*(np.random.rand(H)-0.5)/100    ## random uniform in [-0.01,0.01]

  ## predicted output
  yy=np.zeros(m+1)

  irandom=0    ## =1 random sample, =0 iteration on the input 
  
  for t in range(niteration):
    xi=np.arange(m) ## indexes of the points
    if irandom==1:
      ## process examples in a random order
      np.random.shuffle(xi) 

    for ix in range(m):
      ## draw a training example
      i=xi[ix]   ## read the indexes 
      x=xx[i]    ## input
      r=rr[i]    ## output
      ## forward propagation
      zlin=x*W  
      ## apply sigmoid activation function 
      z=1/(1+np.exp(-zlin))
      ## compute prediction       )
      y=np.dot(V,z)
      ## store the prediction
      yy[i]=y
      ## Backpropagation
      ## change of the weights between the hidden and the output layers
      dV=eta*(r-y)*z
      ## change of the weights between the input and the hidden layers
      dW=eta*(r-y)*V*z*(1-z)*x
      ## update the weights
      V=V+dV
      W=W+dW


  print('Error, r-y, at x=0:',rr[m0+1]-yy[m0+1])

    

## ###################################################
## ################################################################
if __name__ == "__main__":
  main()
          
## ####################################################
 
