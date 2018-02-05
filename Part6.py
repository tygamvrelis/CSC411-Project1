## Part6.py
# In this file, functions are defined to enable training using one-hot label
# encoding. The machinery for part 7 is also set up.

import numpy as np
import math
import os
import matplotlib.pyplot as plt

def part6_J(theta, X, Y):
    '''
    part6_J returns the cost associated with using the (n x k) parameter matrix 
    theta to fit the data X to the corresponding labels Y.
    
    Arguments:
        theta -- (n x k) matrix of learned parameters
        x -- (n x m) matrix whose columns correspond to images (data from which
             to make predictions)
        Y -- (k x m) matrix whose columns corrspond to the actual/target outputs
             (labels)
    '''
    
    return np.sum(np.square(np.dot(theta.T, X) - Y))
    
def part6_grad_J(theta, X, Y):
    '''
    part6_grad_J returns the (n x k) derivative matrix of the cost function J 
    defined in part6_J with respect to the learned parameter matrix theta.
    
    Arguments:
        theta -- (n x k) matrix of learned parameters
        x -- (n x m) matrix whose columns correspond to images (data from which
             to make predictions)
        Y -- (k x m) matrix whose columns corrspond to the actual/target outputs
             (labels)
    '''
    
    return 2 * np.dot(X, np.transpose((np.dot(theta.T, X) - Y)))
    
def part6_grad_J_finite_diff(theta, X, Y, p, q, h):
    '''
    part6_grad_J_finite_diff returns a finite difference approximation of the
    gradient of the cost function define in part6_J.
    
    Arguments:
        theta -- (n x k) matrix of learned parameters
        x -- (n x m) matrix whose columns correspond to images (data from which
             to make predictions)
        Y -- (k x m) matrix whose columns corrspond to the actual/target outputs
             (labels)
        p -- the row of the theta matrix whose q-th entry will be adjusted
        q -- the column of the theta matrix whose p-th entry will be adjusted
        h -- the differential quantity
    '''

    # Idea: increase the pq-th entry of the theta matrix by a small amount h,
    # then evaluate J at this theta. From this quantity, subtract J evaluated
    # at the original theta, then divide that difference by the differential
    # quantity h. This will give the partial derivative of J with respect
    # to the pq-th entry of the theta matrix.

    new_theta = theta.copy()
    new_theta[p, q] += h
    return (part6_J(new_theta, X, Y) - part6_J(theta, X, Y)) / h

def part6_gradient_descent(X, Y, init_theta, alpha, eps, max_iter):
    '''
    part6_gradient_descent finds a local minimum of the hyperplane defined by
    the hypothesis dot(theta.T, X). The algorithm terminates when successive
    values of theta differ by less than eps (convergence), or when the number of
    iterations exceeds max_iter.
    
    Arguments:
        X -- input data for X (the data to be used to make predictions)
        Y -- input data for X (the actual/target data)
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
    
    m = Y.shape[1]
    
    # Do-while...
    while(firstPass or
            (np.linalg.norm(current_theta  - previous_theta) > eps and 
            iter < max_iter)):
        firstPass = False
        
        previous_theta = current_theta.copy() # Update the previous theta value
        
        # Update theta
        current_theta = current_theta - alpha * part6_grad_J(current_theta, X, Y)
        
        if(iter % (max_iter // 100) == 0):
            # Print updates every so often
            cost = part6_J(current_theta, X, Y)
            history.append((iter, cost))
            print("Iter: ", iter, " | Cost: ", cost)
            
        iter += 1
    
    return(current_theta, history)

def part6_classifier(theta, x):
    '''
    part6_classifier returns a (6 x 1) vector, where the i-th entry indicates
    the probability of the image x being the i-th actor. The actor coordinates
    are defined as follows:
        [1 0 0 0 0 0]^T <--> bracco
        [0 1 0 0 0 0]^T <--> gilpin
        [0 0 1 0 0 0]^T <--> harmon
        [0 0 0 1 0 0]^T <--> baldwin
        [0 0 0 0 1 0]^T <--> hader
        [0 0 0 0 0 1]^T <--> carell
    Note that the concatenation of the above forms an identity matrix.
    
    Arguments:
        theta -- the matrix of learned parameters
        x -- the input image (flattened vector with normalized entries)
    '''
    
    pred = np.dot(theta.T, x)
    max_index = np.argmax(pred)
    pred = np.zeros(shape = pred.shape)
    pred[max_index,:] = 1
    return pred
    
    
def part6_imageListToMatrix(input):
    '''
    part6_imageListToMatrix returns tuple (matrix, label) created from a list
    of matrices and their labels
    
    Arguments:
        input -- a list with entries of the form (imageMatrix, labels)
    '''
    
    num_rows = input[0][0].shape[0] # for one actor image matrix
    num_columns = input[0][0].shape[1] # for one actor image matrix
    num_rows_y = input[0][1].shape[0] # for labels
    
    x = np.zeros(shape = (num_rows, num_columns * len(input)))
    y = np.zeros(shape = (num_rows_y, num_columns * len(input)))
    
    j = 0
    for a in input:
        for i in range(num_columns):
            x[:, i + j * num_columns] = a[0][:, i] # data to predict upon (images)
            y[:, i + j * num_columns] = a[1][:, i] # target/actual values (labels)
        j += 1
    
    return (x, y) # x is the matrix of images, y is the label matrix

def part6_getImagesAndLabels(name, directory, label, num_bins):
    '''
    part6_getImagesAndLabels returns a tuple containing all the images
    corresponding to name in the specified directory, as well as the label
    vector for the matrix.
    
    Arguments:
        name -- the string corresponding to the actor name (e.g. "baldwin")
        directory -- the string specifying the direcctory to retrieve the images
                     from (e.g. "training")
        label -- the one-hot label for the images (e.g. 1, 2, 3, 4, 5, 6, etc.).
                 A label of 1 will place a 1 in the first row of the label
                 vector and zeros elsewhere.
        num_bins -- the number of rows for the label vector
    '''
    
    pool = os.listdir(directory)
    imageList = [image for image in pool if name in image]
    
    matrix = np.zeros(shape = (32 * 32, len(imageList)))
    for i in range(len(imageList)):
        currentImage = plt.imread(directory + "/" + imageList[i]).flatten()
        matrix[:,i] = currentImage / 255 # normalize entires
        
    labels = np.zeros((num_bins, len(imageList)))
    labels[label - 1,:] = 1
    
    return(matrix, labels)
    
def part6_getActorLists():
    '''
    part6_getActorLists returns a 4-tuple consisting of the various
    training and validation images and their labels.
    
    e.g. actList = [(braccoMatrix, braccoLabels), (gilpinMatrix, gilpinLabels),
                    ..., (carellMatrix, carellLabels)]
    
    Note that the matrices have the x_0 = 1 terms stacked onto them already by
    the time they are returned in the lists.
    
    Arguments:
        None
    '''
    
    num_bins = 6
    
    # Load images for each actor into numpy arrays and generate an array for the
    # labels associated with each actor's gender.
    directory = "training"
    actList = list()
    actList.append(part6_getImagesAndLabels("bracco", directory, 1, num_bins))
    actList.append(part6_getImagesAndLabels("gilpin", directory, 2, num_bins))
    actList.append(part6_getImagesAndLabels("harmon", directory, 3, num_bins))
    actList.append(part6_getImagesAndLabels("baldwin", directory, 4, num_bins))
    actList.append(part6_getImagesAndLabels("hader", directory, 5, num_bins))
    actList.append(part6_getImagesAndLabels("carell", directory, 6, num_bins))
    actList = [(np.vstack((np.ones((1, a[0].shape[1])), a[0])), a[1])
                     for a in actList]
        
    # Load arrays for validation from act now as well
    directory = "validation"
    val_actList = list()
    val_actList.append(part6_getImagesAndLabels("bracco", directory, 1, num_bins))
    val_actList.append(part6_getImagesAndLabels("gilpin", directory, 2, num_bins))
    val_actList.append(part6_getImagesAndLabels("harmon", directory, 3, num_bins))
    val_actList.append(part6_getImagesAndLabels("baldwin", directory, 4, num_bins))
    val_actList.append(part6_getImagesAndLabels("hader", directory, 5, num_bins))
    val_actList.append(part6_getImagesAndLabels("carell", directory, 6, num_bins))
    val_actList = [(np.vstack((np.ones((1, a[0].shape[1])), a[0])), a[1])
                     for a in val_actList]
    
    (actMatrix, actLabels) = part6_imageListToMatrix(actList)
    (val_actMatrix, val_actLabels) = part6_imageListToMatrix(val_actList)
    
    return(actMatrix, actLabels, val_actMatrix, val_actLabels)