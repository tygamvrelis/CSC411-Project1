## Part3.py
# In this file, functions are defined that help build a classifier to
# distinguish pictures of Alec Baldwin from Steve Carell using linear
# regression. The cost function is optimized using gradient descent.
    
import numpy as np
import math
import os
import matplotlib.pyplot as plt
    
def part3_gradient_descent(x, y, init_theta, alpha, eps, max_iter):
    '''
    part3_gradient_descent finds a local minimum of the hyperplane defined by
    the hypothesis dot(theta.transpose(), x). The algorithm terminates when 
    successive values of theta differ by less than eps (convergence), or when 
    the number of iterations exceeds max_iter.
    
    Arguments:
        x -- input data for x (the data to be used to make predictions)
        y -- input data for y (the actual/target data)
        init_theta -- the initial guess for the local minimum (starting point)
        alpha -- the learning rate; proportional to the step size
        eps -- used to determine img[0]when the algorithm has converged on a 
               solution
        max_iter -- the maximum number of times the algorithm will loop before
                    terminating
    '''
    iter = 0
    previous_theta = 0
    current_theta = init_theta.copy()
    firstPass = True
    history = list()
    
    m = len(y)
    
    # Do-while...
    while(firstPass or
            (np.linalg.norm(current_theta  - previous_theta) > eps and 
            iter < max_iter)):
        firstPass = False
        
        previous_theta = current_theta.copy() # Update the previous theta value
        loss = np.dot(current_theta.T, x) - y # Loss associated with current_theta
        
        # np.dot(x, loss.T) / m is the average gradient of cost
        current_theta = current_theta - alpha * np.dot(x, loss.T) / m # Update theta
        
        if(iter % (max_iter // 100) == 0):
            # Print updates every so often
            cost = np.sum(loss ** 2) / (2 * m) # Compute cost
            cost = math.sqrt(cost)
            history.append((iter, cost))
            print("Iter: ", iter, " | Cost: ", cost)
            
        iter += 1
    
    return(current_theta, history)
        

def part3_classifier(theta, x):
    '''
    part3_classifier returns 1 when x is hypothesized to be an image of Steve 
    Carell, and 0 when x is hypothesized to be an image of Alec Baldwin. 
    
    Arguments:
        theta -- the vector of learned parameters
        x -- the input image (flattened vector with normalized entries)
    '''
    
    return 1 if np.dot(theta.T, x) >= 0.5 else 0
    
def getImageMatrixAndLabels(name, directory, label):
    '''
    getImageMatrixAndLabels returns a tuple containing all the images
    corresponding to name in the specified directory, as well as the label
    vector for the matrix.
    
    Arguments:
        name -- the string corresponding to the actor name (e.g. "baldwin")
        directory -- the string specifying the direcctory to retrieve the images
                     from (e.g. "training")
        label -- the label for the images (e.g. 1, 0, -1, etc...)
    '''
    
    pool = os.listdir(directory)
    imageList = [image for image in pool if name in image]
    
    matrix = np.zeros(shape = (32 * 32, len(imageList)))
    for i in range(len(imageList)):
        currentImage = plt.imread(directory + "/" + imageList[i]).flatten()
        matrix[:,i] = currentImage / 255 # normalize entires
        
    labels = np.full((1, len(imageList)), label)
    
    return(matrix, labels)