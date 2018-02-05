
def project1():
    '''Gets the FaceScrub dataset and performs various analyses on it.
    Returns:
    none
    
    Arguments:
    none
    '''
    
    ## Input
    getInput = 1 # flag to perform ALL input processing
    
    # The input: make directories for the images
    try:
        os.makedirs("uncropped")
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    try:
        os.makedirs("cropped")
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    
    # The input: download the images
    doImageGet = 0
    if(doImageGet or getInput):
        gd.downloadImages("datasets/female_faces.txt")
        gd.downloadImages("datasets/male_faces.txt")
    
    # The input: remove images that aren't valid image files (also removes .png)
    doImageFilter = 0
    if(doImageFilter or getInput):
        pool = os.listdir("uncropped")
        bad = [image for image in pool if imghdr.what("uncropped/" + image) != 'jpeg']
        for image in bad:
            os.remove("uncropped/" + image)
        
    # The input: crop images
    doImageCrop = 0
    if(doImageCrop or getInput):
        gd.cropImages("datasets/female_faces.txt")
        gd.cropImages("datasets/male_faces.txt")
        
    # The input: convert images to greyscale
    doGreyscale = 0
    if(doGreyscale or getInput):
        gd.makeGreyScale("datasets/female_faces.txt", "cropped/")
        gd.makeGreyScale("datasets/male_faces.txt", "cropped/")
        
    # The input: resize images to 32 x 32
    doResize = 0
    if(doResize or getInput):
        gd.resizeImages("datasets/female_faces.txt", "cropped/")
        gd.resizeImages("datasets/male_faces.txt", "cropped/")
        
    ## Part 1: Describe the dataset of faces
    rand.seed(3)
    actorNames = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
    lastNames = list()
    for a in actorNames:
        lastNames.append(a.split(" ")[1])
        
    pool = os.listdir("uncropped")
    exampleFaces = list()
    for i in range(3):
        exampleFaces.append(pool[math.floor(rand.random()*(len(pool) - 1))])
    
    for i in range(3):
        fig = plt.figure(i + 1)
        img = imread("uncropped/" + exampleFaces[i])
        plt.imsave(fname = imagePath + "p1_" + exampleFaces[i][:-4] + "_uncropped" + ".png",
                   arr = img
                   )
        plt.gcf().clear()
    
    for i in range(3):
        fig = plt.figure(i + 4)
        img = imread("cropped/" + exampleFaces[i])
        plt.imshow(img, cmap = plt.cm.gray)
        plt.imsave(fname = imagePath + "p1_" + exampleFaces[i][:-4] + "_cropped" + ".png",
                   arr = img,
                   cmap = plt.cm.gray
                   )
        plt.gcf().clear()
        
    ## Part 2:  Create training set (70-100 images per actor),
    #           validation set (10 images per actor),
    #           and test set (10 images per actor)
    #           These are all non-overlapping (no images in common)

    rand.seed(3)
    # Create all the sets as empty lists
    trainingSet = list()
    validationSet = list()
    testSet = list()
    
    actorNames = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
    lastNames = list()
    for a in actorNames:
        lastNames.append(a.split(" ")[1].lower())
    lastNames = lastNames + ['ferrera', 'drescher', 'chenoweth', 'radcliffe', 'butler', 'vartan']
    # Create new directories for the various sets of images
    try:
        os.makedirs("training")
        os.makedirs("validation")
        os.makedirs("test")
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    
    # Get the set of files names to select from for the other sets. We choose
    # to list the files from the "cropped" relative directory because some
    # images that were downloaded have invalid formats. The images that survive
    # the "cut" are placed in the "cropped" directory. We then select from it
    # without replacement
    pool = os.listdir("cropped")
    
    # Training set creation (70 images per actor)
    # "For each actor whose last name is listed in lastNames..."
    for actor in lastNames:
        # Create a new list containing just the names of the images in pool
        # for the current actor
        actorImages = [image for image in pool if actor.lower() in image]
        for i in range(70):
            # Get a random index for the actor, corresponding to an image in
            # the actorImages list
            idx = round(rand.random()*(len(actorImages) - 1))
            
            # Add the image name to the training set
            trainingSet.append(actorImages[idx])
            
            # Remove the image name from the pool (so that it's not selected
            # for the validation or test set), and from the actorImages list
            # (so that it's not selected again for the training set)
            pool.remove(actorImages[idx])
            del actorImages[idx]
                
    
    # Validation set creation (10 images per actor)
    for actor in lastNames:
        # Create a new list containing just the names of the images in pool
        # for the current actor
        actorImages = [image for image in pool if actor.lower() in image]
        for i in range(10):
            # Get a random index for the actor, corresponding to an image in
            # the actorImages list
            idx = round(rand.random()*(len(actorImages) - 1))
            
            # Add the image name to the validation set
            validationSet.append(actorImages[idx])
            
            # Remove the image name from the pool (so that it's not selected
            # for the test set), and from the actorImages list (so that it's not
            # selected again for the validation set)
            pool.remove(actorImages[idx])
            del actorImages[idx]
    
    # Test set creation (10 images per actor)
    for actor in lastNames:
        # Create a new list containing just the names of the images in pool
        # for the current actor
        actorImages = [image for image in pool if actor.lower() in image]
        for i in range(10):
            # Get a random index for the actor, corresponding to an image in
            # the actorImages list
            idx = round(rand.random()*(len(actorImages) - 1))
            
            # Add the image name to the test set
            testSet.append(actorImages[idx])
            
            # Remove the image name from the actorImages list so that it's not
            # selected again for the validation set
            del actorImages[idx]
            
    # Place the images in the directories created
    for i in range(len(trainingSet)):
        shutil.copy2("cropped/" + trainingSet[i], "training/")
    
    for i in range(len(validationSet)):
        shutil.copy2("cropped/" + validationSet[i], "validation/")
        
    for i in range(len(testSet)):
        shutil.copy2("cropped/" + testSet[i], "test/")
        
    ## Part 3: Classifier for Alec Baldwin and Steve Carell using linear regression
    # Load each training image into a feature matrix as a flattened vector
    #
    # Recovery in 2D array: plt.imshow(baldwinMatrix[:,69].reshape((32,32)))
    rand.seed(3)
    (baldwinMatrix, baldwinLabels) = p3.getImageMatrixAndLabels("baldwin", "training", 1)
    (carellMatrix, carellLabels) = p3.getImageMatrixAndLabels("carell", "training", 0)
    
    # We now begin preparations for gradient descent to optimize theta
    # Notes:
    #  - alpha too large will cause the algorithm to diverge to infinity
    #  - I decreased alpha such that the algorithm would converge to a local
    #    minima that I had already observed within 300000 iterations
    #  - I then tweaked epsilon until this local minimum was converged to
    #  - I then tested the classifier performance on the validation set, and
    #    decreased epsilon from 4E-6 to 1E-6. This change made me go from 2/20
    #    incorrectly classified images to 1/20.
    p3_x = np.hstack((baldwinMatrix, carellMatrix))
    # x_0 is always equal to 1, so we add this "ontop" of the vector x
    p3_x = np.vstack((np.ones((1, p3_x.shape[1])), p3_x))   # Data to be used as guesses
    p3_y = np.hstack((baldwinLabels, carellLabels))         # Actual (target) data
    p3_init_theta = np.zeros(shape = (p3_x.shape[0],1))      # Initial theta parameters
    p3_alpha = 5E-7                                     # Learning rate
    p3_eps = 1E-6                                           # Convergence criterion
    p3_max_iter = 300000                                    # Maximum number of iterations
    
    # Gradient descent
    (theta, history) = p3.part3_gradient_descent(p3_x, p3_y, p3_init_theta, p3_alpha, 
                                      p3_eps, p3_max_iter)
    
    # Plot cost as a function of time
    x_values = [i[0] for i in history]
    y_values = [i[1] for i in history]
    plt.plot(x_values, y_values)
    plt.title("(Part 3) Cost function minimization using gradient descent")
    plt.xlabel("Number of iterations")
    plt.ylabel("Cost")
    plt.yscale('log')
    plt.savefig(imagePath + "p3_gradient_descent.jpg")
    plt.gcf().clear()
    
    # Test performance of the classifier on the training set
    countTraining = 0
    costTraining = 0
    for i in range(p3_x.shape[1]):
        countTraining += abs(p3_y.item(i) - p3.part3_classifier(theta, p3_x[:,i:i + 1]))
        costTraining += (p3_y.item(i) - np.dot(theta.T, p3_x[:,i:i + 1])) ** 2
        # print(p3.part3_classifier(theta, p3_x[:,i:i + 1]))
    print("(Training): Percentage of correct classifications: ", 1 - (countTraining / p3_x.shape[1]))
    print("(Training): Average cost: ", costTraining.item(0) / p3_x.shape[1])
        
    # Test performance of the classifier on the validation set
    # Load validation images now as well
    (valBaldwinMatrix, valBaldwinLabels) = p3.getImageMatrixAndLabels("baldwin", "validation", 1)
    (valCarellMatrix, valCarellLabels) = p3.getImageMatrixAndLabels("carell", "validation", 0)
    
    val_p3_x = np.hstack((valBaldwinMatrix, valCarellMatrix))
    val_p3_x = np.vstack((np.ones((1, val_p3_x.shape[1])), val_p3_x))
    val_p3_y = np.hstack((valBaldwinLabels, valCarellLabels))   
    
    countValidation = 0
    costValidation = 0
    for i in range(val_p3_x.shape[1]):
        countValidation += abs(val_p3_y.item(i) - p3.part3_classifier(theta, val_p3_x[:,i:i + 1]))
        costValidation += (val_p3_y.item(i) - np.dot(theta.T, val_p3_x[:,i:i + 1])) ** 2
        # print(p3.part3_classifier(theta, p3_x[:,i:i + 1]))
    print("(Validation): Percentage of correct classifications: ",
          (1 - countValidation / val_p3_x.shape[1]))
    print("(Validation): Average cost: ",
          costValidation.item(0) / val_p3_x.shape[1])

    ## Part 4: Displaying the thetas as an image
    # Part (a.1): thetas obtained using the full training set
    # Note: run part 3 before this part
    plt.yscale('linear')
    plt.title("Part 4 (a): Theta visualization using entire training set")
    plt.imshow(theta[1:].reshape((32,32)), interpolation = 'gaussian', cmap = plt.cm.coolwarm)
    plt.savefig(imagePath + "p4_a_thetas_full_set.jpg")
    plt.gcf().clear()
    
    # Part (a.2): thetas obtained using only 2 images from each actor
    p4_x = np.hstack((baldwinMatrix[:,0:2], carellMatrix[:,0:2]))
    p4_x = np.vstack((np.ones((1, p4_x.shape[1])), p4_x))
    p4_y = np.hstack((baldwinLabels[:,0:2], carellLabels[:,0:2]))   
    p4_init_theta = np.zeros(shape = (p4_x.shape[0],1))
    (theta, history) = p3.part3_gradient_descent(p4_x, p4_y, p4_init_theta, 5E-6, 
                                      2E-6, 300000)
    plt.yscale('linear')
    plt.title("Part 4 (a): Theta visualization using 2 images for each actor")
    plt.imshow(theta[1:].reshape((32,32)), interpolation = 'gaussian', cmap = plt.cm.coolwarm)
    plt.savefig(imagePath + "p4_a_thetas_2_actor_images.jpg")
    plt.gcf().clear()
    
    # Part (b): Using different strategies to obtain thetas that look like a
    #           face and don't look like a face using the FULL training set
     
    # First, the thetas that don't look like a face. These are obtained by
    # forcing epsilon to be small, thus forcing theta to be more closely fit to
    # all the features in the images
    p4b_init_theta = np.zeros(shape = (p3_x.shape[0],1))
    (theta, history) = p3.part3_gradient_descent(p3_x, p3_y, p4b_init_theta, 5E-6, 
                                      2E-6, 300000)
    plt.yscale('linear')
    plt.title("Part 4 (b): Thetas (eps = 2E-6, alpha = 5E-6)")
    plt.imshow(theta[1:].reshape((32,32)), interpolation = 'gaussian', cmap = plt.cm.coolwarm)
    plt.savefig(imagePath + "p4_b_thetas_no_face.jpg")
    plt.gcf().clear()
    
    # Next, we make the thetas come out like a face (epsilon larger)
    (theta, history) = p3.part3_gradient_descent(p3_x, p3_y, p4_init_theta, 5E-6, 
                                      1E-3, 300000)
    plt.yscale('linear')
    plt.title("Part 4 (b): Thetas (eps = 1E-3, alpha = 5E-6)")
    plt.imshow(theta[1:].reshape((32,32)), interpolation = 'gaussian', cmap = plt.cm.coolwarm)
    plt.savefig(imagePath + "p4_b_thetas_face.jpg")
    plt.gcf().clear()
    
    ## Part 5: Classifying actors as male or female & determining the 
    ##         relationship between training set size and classifier performance
    #   This part has three related subparts:
    #       1. Obtain the relationship between the size of the training set, and
    #          the classifier performance on:
    #               1.1 the training set
    #               1.2 the validation set consisting of the actors in act
    #       2. Obtain the relationship between the size of the training set, and
    #          the classifier performance on:
    #               2.1 the validation set consisting of the 6 actors NOT in act
    #       3. Complete the previous 2 subparts in such a way that overfitting
    #          is demonstrated

    (actList, val_actList, val_not_actList) = p5.part5_getActorLists()
    
    # Train using actList. Try to get very close to the local minimum since we
    # want to demonstrate overfitting
    rand.seed(3)
    p5_history = list()
    for i in range(70):
        print("PART 5: i = ", i)
        (theta, history) = p5.part5_train(actList, i + 1, 1.60E-5, 5E-7, 600000)
        (cost_actList, corr_actList) = p5.part5_classify(actList, theta, i + 1) # Test on training set
        if(i < 60):
            (cost_val_actList, corr_val_actList) = p5.part5_classify(val_actList, theta, i + 1) # Test on validation set
            (cost_val_not_actList, corr_val_not_actList) = p5.part5_classify(val_not_actList, theta, i + 1) # Test on validation set for 6 actors not in act
        else:
            (cost_val_actList, corr_val_actList) = p5.part5_classify(val_actList, theta, 60) # Test on validation set
            (cost_val_not_actList, corr_val_not_actList) = p5.part5_classify(val_not_actList, theta, 60) # Test on validation set for 6 actors not in act
        
        p5_history.append((cost_actList, corr_actList, cost_val_actList, corr_val_actList,
                           cost_val_not_actList, corr_val_not_actList))
                           
    # Plot cost as a function of training set size
    t = np.arange(1, 71, 1)
    y_cost_actList = [i[0] for i in p5_history]
    y_cost_val_actList = [i[2] for i in p5_history]
    y_cost_val_not_actList = [i[4] for i in p5_history]
    plt.plot(t, y_cost_actList, 'r--', label = 'Training set')
    plt.plot(t, y_cost_val_actList, 'g--', label = 'Validation set V1')
    plt.plot(t, y_cost_val_not_actList, 'b--', label = 'Validation set V2')
    plt.title("(Part 5) Average prediction cost as a function of training set size")
    plt.xlabel("Number of images per actor in training set")
    plt.ylabel("Cost (average)")
    plt.yscale('log')
    plt.legend()
    plt.savefig(imagePath + "p5_cost.jpg")
    plt.gcf().clear()
    
    # Plot percentage of correct classifications as a function of training set size
    t = np.arange(1, 71, 1)
    y_corr_actList = [i[1] for i in p5_history]
    y_corr_val_actList = [i[3] for i in p5_history]
    y_corr_val_not_actList = [i[5] for i in p5_history]
    plt.plot(t, y_corr_actList, 'r--', label = 'Training set')
    plt.plot(t, y_corr_val_actList, 'g--', label = 'Validation set V1')
    plt.plot(t, y_corr_val_not_actList, 'b--', label = 'Validation set V2')
    plt.title("(Part 5) Classification performance as a function of training set size")
    plt.xlabel("Number of images per actor in training set")
    plt.ylabel("Correct classifications (%)")
    plt.yscale('linear')
    plt.legend()
    plt.savefig(imagePath + "p5_correct_percent.jpg")
    plt.gcf().clear()
    
    # print("(Training) Avg cost: ", avg_cost, " | Percent correct: ", perc_correct)
    # print("(Validation) Avg cost: ", val_avg_cost, " | Percent correct: ", val_perc_correct)

    ## Part 6: Using one-hot encoding for inputs
    # Part 6a: computing frac{\partial J}{\partial theta_{pq}} (see report for 
    # derivation)
    
    # Part 6b: show that the derivative of J(\theta) with respect to all the
    # components of \theta can be written in matrix form as 2X(\theta^T X - Y)^T
    # (see report for proof)
    
    # Part 6c: Implement the cost function from part 6a and its vectorized
    # gradient function in Python (code included in report). See Part6.py
    
    # Part 6d: Demonstrate that the vectorized gradient descent function works
    # by computing several components of the gradient using finite-difference
    # approximations. See Part6.py for the helpfunction implementations.
    
    # Load the image data into numpy arrays
    (actMatrix, actLabels, val_actMatrix, val_actLabels) = p6.part6_getActorLists()
    
    # Initialize a dummy theta array (all entries = 0.5)
    dummy_theta = np.full(shape = (actMatrix.shape[0], actLabels.shape[0]), 
                          fill_value = 0.5)
                            
    # Compute the derivative matrix for the cost. Store results in vcomp
    vcomp = p6.part6_grad_J(dummy_theta, actMatrix, actLabels)
    
    # Select 5 random entries in the derivative matrix to compute using the
    # finite difference approximation. Print the entry index, the finite
    # difference approximation, and the vectorized computation.
    rand.seed(3)
    error = 0
    for i in range(5):
        p = round(actMatrix.shape[0] * rand.random())
        q = round(actLabels.shape[0] * rand.random())
        apprx = p6.part6_grad_J_finite_diff(dummy_theta, actMatrix, actLabels, 
                                            p, q, 1E-5)
        error += abs(vcomp[p, q] - apprx) / vcomp[p, q]
        print("p: ", p, "| q: ", q, "| diff. apprx: ", apprx,
              "| vectorized comp. : ", vcomp[p, q])
    print("Avg. error: ", error / 5 * 100) 

    ## Part 7: Face recognition for 6 actors using one-hot label encoding
    # Load the image data into numpy arrays
    (actMatrix, actLabels, val_actMatrix, val_actLabels) = p6.part6_getActorLists()
    
    # Initialize an initial theta guess
    init_theta = np.full(shape = (actMatrix.shape[0], actLabels.shape[0]), 
                          fill_value = 0.1)
                          
    # Gradient descent
    alpha = 1E-6
    eps = 6E-5
    (theta, history) = p6.part6_gradient_descent(actMatrix, actLabels, init_theta, alpha,
                                      eps, 15000)
    
    # Test performance of the classifier on the training set
    countTraining = 0
    costTraining = 0
    for i in range(actMatrix.shape[1]):
        countTraining += np.dot(actLabels[:, i],
                             p6.part6_classifier(theta, actMatrix[:,i:i + 1]))
        costTraining += np.sum((actLabels[:, i] - 
                                np.dot(theta.T, actMatrix[:,i:i + 1])) ** 2)
        # print(p3.part3_classifier(theta, actMatrix[:,i:i + 1]))
    print("(Training): Percentage of correct classifications: ", countTraining / actMatrix.shape[1])
    print("(Training): Average cost: ", costTraining.item(0) / actMatrix.shape[1])
    
    # Test performance of the classifier on the validation set
    countTraining = 0
    costTraining = 0
    for i in range(val_actMatrix.shape[1]):
        countTraining += np.dot(val_actLabels[:, i],
                             p6.part6_classifier(theta, val_actMatrix[:,i:i + 1]))
        costTraining += np.sum((val_actLabels[:, i] - 
                                np.dot(theta.T, val_actMatrix[:,i:i + 1])) ** 2)
        # print(p3.part3_classifier(theta, val_actMatrix[:,i:i + 1]))
    print("(Validation): Percentage of correct classifications: ", countTraining / val_actMatrix.shape[1])
    print("(Validation): Average cost: ", costTraining.item(0) / val_actMatrix.shape[1])
    
    ## Part 8: Visualization of learned parameter vector theta from part 7
    # Note: run part 7 before running this part

    key = [(1, "bracco"), (2, "gilpin"), (3, "harmon"), (4, "baldwin"), (5, "hader"), (6, "carell")]
    
    for k in key:
        plt.yscale('linear')
        plt.title("(Part 8) " + k[1])
        plt.imshow(theta[1:,k[0] - 1].reshape((32,32)), interpolation = 'gaussian', cmap = plt.cm.coolwarm)
        plt.savefig(imagePath + "p8_" + k[1] + ".jpg")
        plt.gcf().clear()

if __name__ == "__main__":
    # Import libraries
    import numpy as np
    import numpy.random as rand
    import matplotlib.pyplot as plt
    import matplotlib.cbook as cbook
    import time
    from scipy.misc.pilutil import imread
    from scipy.misc.pilutil import imresize
    from scipy.misc import imsave
    import matplotlib.image as mpimg
    import os
    import shutil
    import errno
    from scipy.ndimage import filters
    import urllib.request
    import math
    import imghdr # for removing invalid images

    os.chdir("D:/Users/Tyler/Documents/tyler/School/University of Toronto/Year 3/CSC411-MachineLearningAndDataMining/Project1")
    
    # Import local files
    import get_data as gd
    import Part3 as p3
    import Part5 as p5
    import Part6 as p6
    
    # Set path for all images to be saved into
    try:
        os.makedirs("Report/Images/")
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    imagePath = "Report/Images/"
    
    # Seed the generator to get identical results each time
    rand.seed(3)
    
    project1()