
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc.pilutil import imread
from scipy.misc.pilutil import imresize
from scipy.misc import imsave
import matplotlib.image as mpimg
import os
import errno
from scipy.ndimage import filters
import urllib.request

def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/'''
    
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result  

def downloadImages(dataset):
    '''Downloads the images from the dataset into os.getcwd()/uncropped.
    
    Returns:
    none
    
    Arguments:
    dataset -- a string representing the relative path of the dataset
    '''
    
    act = list(set([a.split("\t")[0] for a in open(dataset).readlines()]))
    for a in act:
        name = a.split()[1].lower()
        i = 0
        for line in open(dataset):
            if a in line:
                filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
                
                if not os.path.isfile("uncropped/"+filename):
                    if(timeout(urllib.request.urlretrieve, (line.split()[4], "uncropped/"+filename), {}, 45) == False):
                        print("FAILED to downloaded image: " + filename)
                    else:
                        print("Downloaded image: " + filename)
                else:
                    print("Skipped (already downloaded): " + filename)
                    
                i += 1
                
                
def cropImages(dataset):
    '''Crops the images in the uncropped folder and places the results in the 
    cropped folder (both relative to the current working directory).
    
    Returns:
    none
    
    Arguments:
    dataset -- a string representing the relative path of the dataset
    '''
    
    act = list(set([a.split("\t")[0] for a in open(dataset).readlines()]))
    for a in act:
        name = a.split()[1].lower()
        i = 0
        for line in open(dataset):
            if a in line:
                filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
                if(os.path.isfile("uncropped/"+filename)):
                    if not os.path.isfile("cropped/"+filename):
                        boundingBox = line.split("\t")[4].split(",") # Get coordinates of bounding box
                        x1 = int(boundingBox[0])
                        y1 = int(boundingBox[1])
                        x2 = int(boundingBox[2])
                        y2 = int(boundingBox[3])
                        
                        try:
                            img = imread("uncropped/"+filename) # Read image as a numpy array
                            img = img[y1:y2, x1:x2] # Crop the image to the bounding box for the face, specified in the dataset
                            imsave("cropped/"+filename, img) # Save the image
                            print("Cropped image: " + filename)
                        except (OSError, ValueError) as e:
                            print(e)
                    else:
                        print("Skipped (already cropped): " + filename)
                
                i += 1

def rgb2gray(rgb):
    '''Return the grayscale version of the RGB image rgb as a 2D numpy array
    whose range is 0..1
    Arguments:
    rgb -- an RGB image, represented as a numpy array of size n x m x 3. The
    range of the values is 0..255
    '''
    
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray/255.

def makeGreyScale(dataset, directory):
    '''Makes all the images in the directory that are a subset of the dataset
    greyscale.
    
    Returns:
    none
    
    Arguments:
    dataset -- a string representing the relative path of the dataset
    directory -- a string representing the relative path of the images.
                 String should be terminated with a slash; for example: "cropped/"
    '''

    act = list(set([a.split("\t")[0] for a in open(dataset).readlines()]))
    for a in act:
        name = a.split()[1].lower()
        i = 0
        for line in open(dataset):
            if a in line:
                filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
                path = directory + filename
                if(os.path.isfile(path)):
                    try:
                        img = imread(path) # Read image as a numpy array
                        img = rgb2gray(img) # Convert to greyscale
                        imsave(path, img) # Save the image
                        print("Made greyscale: " + filename)
                    except (OSError, IndexError) as e:
                        print(e)
                i += 1
                
def resizeImages(dataset, directory):
    '''Makes all the images in the directory that are a subset of the dataset
    32 x 32 in size.
    
    Returns:
    none
    
    Arguments:
    dataset -- a string representing the relative path of the dataset
    directory -- a string representing the relative path of the images.
                 String should be terminated with a slash; for example: "cropped/"
    '''
    
    act = list(set([a.split("\t")[0] for a in open(dataset).readlines()]))
    for a in act:
        name = a.split()[1].lower()
        i = 0
        for line in open(dataset):
            if a in line:
                filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
                path = directory + filename
                if(os.path.isfile(path)):
                    try:
                        img = imread(path) # Read image as a numpy array
                        img = imresize(img, (32,32)) # Resize image to 32 x 32
                        imsave(path, img) # Save the image
                        print("Resized: " + filename)
                    except (OSError, IndexError) as e:
                        print(e)
                else:
                    print("File not found: " + filename)
                    
                i += 1