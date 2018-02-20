## Part5.py
# In this file, functions are defined to help determine the relationship
# between training set size and classification performance on the training
# and validation sets.

import numpy as np
import numpy.random as rand
import math
import os
import matplotlib.pyplot as plt
import Part3 as p3

def part5_train(trainingSet, size, alpha, eps, max_iter):
    '''
    part5_train returns the parameter theta fit to the data in trainingSet using
    size images per actor in linear regression via gradient descent.
    
    Arguments:
     trainingSet -- a list in the form (imageMatrix, labels) used to train theta
     size -- a numerical value between 1 and 70. Specifies how many images of
             each actor will be used to train
     alpha -- gradient descent "learning rate" parameter (proportional to step
              size)
     eps -- gradient descent parameters (determines how tight the convergence
            criteria are)
    '''
    
    # Input validation
    if size > 70 or size < 1:
        if size > 70:
            size = 70
        else:
            size = 1
    
    # Prepare data
    x = np.zeros(shape = (trainingSet[0][0].shape[0], 
                               size * len(trainingSet)))
    y = np.zeros(shape = (1, size * len(trainingSet)))
    
    j = 0
    for a in trainingSet:
        for i in range(size):
            x[:, i + j * size] = a[0][:, i] # data to predict upon (images)
            y[:, i + j * size] = a[1][:, i] # target/actual values (labels)
        j += 1
    
    # Run gradient descent
    init_theta = np.zeros(shape = (x.shape[0],1))
    return p3.part3_gradient_descent(x, y, init_theta, alpha, eps, max_iter)
    
def part5_classify(input, theta, size):
    '''
    part5_classify returns the average cost and percentage of correct
    classifications for the hypothesis np.dot(theta.T, x), using the learned
    parameters theta and testing the images in the input list against the labels
    in the input list.
    
    Arguments:
        input -- a list in the form (imageMatrix, labels) used to make
                predictions and determine performance characteristics
        theta -- the learned parameters that will be used to make predictions
        size -- the size of the classification set to be tested
    '''
    
    cost = 0
    incorrect = 0
    
    num_rows = input[0][0].shape[0] # for one actor image matrix
    num_columns = input[0][0].shape[1] # for one actor image matrix
    
    x = np.zeros(shape = (num_rows, num_columns * len(input)))
    y = np.zeros(shape = (1, num_columns * len(input)))
    
    j = 0
    for a in input:
        for i in range(num_columns):
            x[:, i + j * num_columns] = a[0][:, i] # data to predict upon (images)
            y[:, i + j * num_columns] = a[1][:, i] # target/actual values (labels)
        j += 1
    
    for i in range(size):
        incorrect += abs(y[:,i:i+1] - np.dot(theta.T, x[:,i:i + 1]))
        cost += (y[:,i:i+1] - np.dot(theta.T, x[:,i:i + 1])) ** 2
        
    avg_cost = cost / size
    perc_correct = (size - incorrect) / size
    
    return(avg_cost.item(0), perc_correct.item(0))
    
    
def part5_getActorLists():
    '''
    part5_getActorLists returns a 3-tuple consisting of the various
    training and validation images and their labels stored as tuples in lists.
    
    e.g. actList = [(braccoMatrix, braccoLabels), (gilpinMatrix, gilpinLabels),
                    ..., (carellMatrix, carellLabels)]
    
    Note that the matrices have the x_0 = 1 terms stacked onto them already by
    the time they are returned in the lists.
    
    Arguments:
        None
    '''
    
    # Load images for each actor into numpy arrays and generate an array for the
    # labels associated with each actor's gender.
    # Let:
    #   1 denote a male actor
    #   0 denote a female actor
    directory = "training"
    actList = list()
    actList.append(p3.getImageMatrixAndLabels("bracco", directory, 0))
    actList.append(p3.getImageMatrixAndLabels("gilpin", directory, 0))
    actList.append(p3.getImageMatrixAndLabels("harmon", directory, 0))
    actList.append(p3.getImageMatrixAndLabels("baldwin", directory, 1))
    actList.append(p3.getImageMatrixAndLabels("hader", directory, 1))
    actList.append(p3.getImageMatrixAndLabels("carell", directory, 1))
    actList = [(np.vstack((np.ones((1, a[0].shape[1])), a[0])), a[1])
                     for a in actList]
        
    
    # Load arrays for validation from act now as well
    directory = "validation"
    val_actList = list()
    val_actList.append(p3.getImageMatrixAndLabels("bracco", directory, 0))
    val_actList.append(p3.getImageMatrixAndLabels("gilpin", directory, 0))
    val_actList.append(p3.getImageMatrixAndLabels("harmon", directory, 0))
    val_actList.append(p3.getImageMatrixAndLabels("baldwin", directory, 1))
    val_actList.append(p3.getImageMatrixAndLabels("hader", directory, 1))
    val_actList.append(p3.getImageMatrixAndLabels("carell", directory, 1))
    val_actList = [(np.vstack((np.ones((1, a[0].shape[1])), a[0])), a[1])
                     for a in val_actList]

                               
    # Load arrays for validation from the actors not in act now
    actorNames = ['ferrera', 'drescher', 'chenoweth', 'radcliffe', 'butler', 'vartan']
    # First 3 are female, last 3 are male
    
    directory = "validation"
    val_not_actList = list()
    val_not_actList.append(p3.getImageMatrixAndLabels("ferrera", directory, 0))
    val_not_actList.append(p3.getImageMatrixAndLabels("drescher", directory, 0))
    val_not_actList.append(p3.getImageMatrixAndLabels("chenoweth", directory, 0))
    val_not_actList.append(p3.getImageMatrixAndLabels("radcliffe", directory, 1))
    val_not_actList.append(p3.getImageMatrixAndLabels("butler", directory, 1))
    val_not_actList.append(p3.getImageMatrixAndLabels("vartan", directory, 1))
    val_not_actList = [(np.vstack((np.ones((1, a[0].shape[1])), a[0])), a[1])
                     for a in val_not_actList]
    
    return(actList, val_actList, val_not_actList)
    
# def part5_getActorMatricesAndLabels():
#     '''
#     part5_getActorMatricesAndLabels returns a 6-tuple consisting of the various
#     training and validation images and their labels in the following form:
#     
#     (actMatrix, actLabels, val_actMatrix, val_actLabels, val_not_actMatrix, 
#     val_not_actLabels), where:
#         - actMatrix corresponds to the training images for the 6 actors in list 
#           act
#         - actLabels corresponds to the training labels (male (1) or female (0)) 
#           for the actors in list act
#         - val_actMatrix corresponds to the validation images for the actors in
#           list act
#         - val_actLabels corresponds to the validation labels for the actors in
#           list act
#         - val_not_actMatrix corresponds to the training images for the 6 actors
#           NOT in list act
#         - val_not_actLabels corresponds to the training labels for the 6 actors
#           NOT in list act
#     
#     Arguments:
#         None
#     '''
#     
#     # Load images for each actor into numpy arrays and generate an array for the
#     # labels associated with each actor's gender.
#     # Let:
#     #   1 denote a male actor
#     #   0 denote a female actor
#     directory = "training"
#     (braccoMatrix, braccoLabels) = p3.getImageMatrixAndLabels("bracco", directory, 0)
#     (gilpinMatrix, gilpinLabels) = p3.getImageMatrixAndLabels("gilpin", directory, 0)
#     (harmonMatrix, harmonLabels) = p3.getImageMatrixAndLabels("harmon", directory, 0)
#     (baldwinMatrix, baldwinLabels) = p3.getImageMatrixAndLabels("baldwin", directory, 1)
#     (haderMatrix, haderLabels) = p3.getImageMatrixAndLabels("hader", directory, 1)
#     (carellMatrix, carellLabels) = p3.getImageMatrixAndLabels("carell", directory, 1)
#     
#     actMatrix = np.hstack((braccoMatrix, gilpinMatrix, harmonMatrix, 
#                            baldwinMatrix, haderMatrix, carellMatrix))
#     actMatrix = np.vstack(((np.ones((1, actMatrix.shape[1]))), actMatrix))
#     actLabels = np.hstack((braccoLabels, gilpinLabels, harmonLabels,
#                            baldwinLabels, haderLabels, carellLabels))
#     
#     # Load arrays for validation from act now as well
#     directory = "validation"
#     (val_braccoMatrix, val_braccoLabels) = p3.getImageMatrixAndLabels("bracco", directory, 0)
#     (val_gilpinMatrix, val_gilpinLabels) = p3.getImageMatrixAndLabels("gilpin", directory, 0)
#     (val_harmonMatrix, val_harmonLabels) = p3.getImageMatrixAndLabels("harmon", directory, 0)
#     (val_baldwinMatrix, val_baldwinLabels) = p3.getImageMatrixAndLabels("baldwin", directory, 1)
#     (val_haderMatrix, val_haderLabels) = p3.getImageMatrixAndLabels("hader", directory, 1)
#     (val_carellMatrix, val_carellLabels) = p3.getImageMatrixAndLabels("carell", directory, 1)
#     
#     val_actMatrix = np.hstack((val_braccoMatrix, val_gilpinMatrix,
#                                val_harmonMatrix, val_baldwinMatrix,
#                                val_haderMatrix, val_carellMatrix))
#     val_actMatrix = np.vstack(((np.ones((1, val_actMatrix.shape[1]))),
#                                val_actMatrix))
#     val_actLabels = np.hstack((val_braccoLabels, val_gilpinLabels, 
#                                val_harmonLabels, val_baldwinLabels,
#                                val_haderLabels, val_carellLabels))
#                                
#     # Load arrays for validation from the actors not in act now
#     actorNames = ['ferrera', 'drescher', 'chenoweth', 'radcliffe', 'butler', 'vartan']
#     # First 3 are female, last 3 are male
#     
#     directory = "validation"
#     (val_ferreraMatrix, val_ferreraLabels) = p3.getImageMatrixAndLabels("ferrera", directory, 0)
#     (val_drescherMatrix, val_drescherLabels) = p3.getImageMatrixAndLabels("drescher", directory, 0)
#     (val_chenowthMatrix, val_chenowethLabels) = p3.getImageMatrixAndLabels("chenoweth", directory, 0)
#     (val_radcliffeMatrix, val_radcliffeLabels) = p3.getImageMatrixAndLabels("radcliffe", directory, 1)
#     (val_butlerMatrix, val_butlerLabels) = p3.getImageMatrixAndLabels("butler", directory, 1)
#     (val_vartanMatrix, val_vartanLabels) = p3.getImageMatrixAndLabels("vartan", directory, 1)
#     
#     val_not_actMatrix = np.hstack((val_ferreraMatrix, val_drescherMatrix,
#                                    val_chenowthMatrix, val_radcliffeMatrix,
#                                    val_butlerMatrix, val_vartanMatrix))
#     val_not_actMatrix = np.vstack(((np.ones((1, val_not_actMatrix.shape[1]))),
#                                   val_not_actMatrix))
#     val_not_actLabels = np.hstack((val_ferreraLabels, val_drescherLabels, 
#                                    val_chenowethLabels, val_radcliffeLabels,
#                                    val_butlerLabels, val_vartanLabels))
#     
#     return(actMatrix, actLabels, val_actMatrix, val_actLabels, 
#            val_not_actMatrix, val_not_actLabels)