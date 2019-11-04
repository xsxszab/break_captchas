import numpy as np
import matplotlib.pyplot as plt
import cv2

def process(img):
    '''process and return single image'''
    img = cv2.resize(img,(120,40),interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((2,2),np.uint8)
    img = cv2.dilate(img,kernel,iterations = 1)
    img = cv2.bilateralFilter(img,9,14,3)
    _, img = cv2.threshold(img,110,20,cv2.THRESH_TRUNC)
    kernel_sharpen = np.array([
        [-1,-1,-1],
        [-1,9,-1],
        [-1,-1,-1]])
    img = cv2.filter2D(img,-1,kernel_sharpen)
    return img


def processImg(img_path=None, save_path=None, name=None, imgs=None, prefix='', method=['write']):
    '''
    universial image processing function, 
    require one RGB format image as input
    '''
    img = cv2.imread(img_path+name)
    img = cv2.resize(img,(120,40),interpolation=cv2.INTER_CUBIC) # make sure all images are of the same size
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # transform RGB images into grayscale images
    kernel = np.ones((2,2),np.uint8)
    img = cv2.dilate(img,kernel,iterations = 1) # apply dilate operation on grayscale images
    img = cv2.bilateralFilter(img,9,14,3) # bilateral filtering
    _, img = cv2.threshold(img,110,20,cv2.THRESH_TRUNC)
    kernel_sharpen = np.array([ # modified laplacian filter, sharpen edges
        [-1,-1,-1],
        [-1,9,-1],
        [-1,-1,-1]])
    img = cv2.filter2D(img,-1,kernel_sharpen) # apply the filter
    if not len(method):
         raise Exception('wrong method')
    if 'write' in method: # write processed image to disk
        cv2.imwrite(save_path+prefix+name, img)
    if 'display' in method: # display processed image
        plt.imshow(img, cmap='gray')
        plt.xticks([])
        plt.yticks([])
    if 'memory' in method: # save processed image to memory
        imgs.append(img)

