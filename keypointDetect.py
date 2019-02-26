import numpy as np
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt

def createGaussianPyramid(im, sigma0=1, 
        k=np.sqrt(2), levels=[-1,0,1,2,3,4]):
    if len(im.shape)==3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if im.max()>10:
        im = np.float32(im)/255
    im_pyramid = []
    for i in levels:
        sigma_ = sigma0*k**i 
        im_pyramid.append(cv2.GaussianBlur(im, (0,0), sigma_))
    im_pyramid = np.stack(im_pyramid, axis=-1)
    return im_pyramid

def displayPyramid(im_pyramid):
    im_pyramid = np.split(im_pyramid, im_pyramid.shape[2], axis=2)
    im_pyramid = np.concatenate(im_pyramid, axis=1)
    im_pyramid = cv2.normalize(im_pyramid, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.imshow('Pyramid of image', im_pyramid)
    cv2.waitKey(0) # press any key to exit
    cv2.destroyAllWindows()

def displayKeypoints(locs, im_pyr, levels=[-1, 0, 1, 2, 3, 4]):
    im = im_pyr[:,:,0]
    im = np.repeat(im.reshape((im.shape[0], im.shape[1], 1)), 3, axis=2)
    fig = plt.figure()
    plt.imshow(im)
    plt.plot(locs[:,0], locs[:,1], 'g.')
    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close(fig)

    for i in range(im_pyr.shape[2]):
        im = im_pyr[:,:,i]
        im = np.repeat(im.reshape((im.shape[0], im.shape[1], 1)), 3, axis=2)
        fig = plt.figure()
        plt.imshow(im)
        for j in range(locs.shape[0]):
            if(locs[j, 2] == levels[i]):
                plt.plot(locs[j,0], locs[j,1], 'g.')
        plt.draw()
        plt.waitforbuttonpress(0)
        plt.close(fig)



def createDoGPyramid(gaussian_pyramid, levels=[-1,0,1,2,3,4]):
    '''
    Produces DoG Pyramid
    Inputs
    Gaussian Pyramid - A matrix of grayscale images of size
                        [imH, imW, len(levels)]
    levels      - the levels of the pyramid where the blur at each level is
                   outputs
    DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
                   created by differencing the Gaussian Pyramid input
    '''
    DoG_pyramid = []
    ################
    # TO DO ...
    # compute DoG_pyramid here
    DoG_levels = levels[1:]
    for level in range(1,gaussian_pyramid.shape[2]):
        diff = gaussian_pyramid[:,:,level] - gaussian_pyramid[:,:,level-1]
        DoG_pyramid.append(diff)
    DoG_pyramid = np.stack(DoG_pyramid, axis=-1)
    return DoG_pyramid, DoG_levels

def computePrincipalCurvature(DoG_pyramid):
    '''
    Takes in DoGPyramid generated in createDoGPyramid and returns
    PrincipalCurvature,a matrix of the same size where each point contains the
    curvature ratio R for the corre-sponding point in the DoG pyramid
    
    INPUTS
        DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
    
    OUTPUTS
        principal_curvature - size (imH, imW, len(levels) - 1) matrix where each 
                          point contains the curvature ratio R for the 
                          corresponding point in the DoG pyramid
    '''
    principal_curvature = []
    ##################
    # TO DO ...
    # Compute principal curvature here
    DoG_pyramid = np.split(DoG_pyramid, DoG_pyramid.shape[2], axis=2)
    for DoG in DoG_pyramid:
        Dxx = cv2.Sobel(DoG,-1, 2, 0, ksize=5)
        Dyy = cv2.Sobel(DoG,-1, 0, 2, ksize=5)
        Dxy = cv2.Sobel(DoG,-1, 1, 1, ksize=5)
        Tr = Dxx + Dyy 
        Det = np.multiply(Dxx,Dyy) - np.multiply(Dxy,Dxy)
        Det[Det==0] = 10000
        R = np.divide(np.multiply(Tr, Tr), Det)
        principal_curvature.append(R)
    principal_curvature = np.stack(principal_curvature, axis=-1)
    return principal_curvature

def getLocalExtrema(DoG_pyramid, DoG_levels, principal_curvature,
        th_contrast=0.03, th_r=12):
    '''
    Returns local extrema points in both scale and space using the DoGPyramid

    INPUTS
        DoG_pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
        DoG_levels  - The levels of the pyramid where the blur at each level is
                      outputs
        principal_curvature - size (imH, imW, len(levels) - 1) matrix contains the
                      curvature ratio R
        th_contrast - remove any point that is a local extremum but does not have a
                      DoG response magnitude above this threshold
        th_r        - remove any edge-like points that have too large a principal
                      curvature ratio
     OUTPUTS
        locsDoG - N x 3 matrix where the DoG pyramid achieves a local extrema in both
               scale and space, and also satisfies the two thresholds.
    '''
    locsDoG = []
    ##############
    #  TO DO ...
    # Compute locsDoG here
    for i in range(DoG_pyramid.shape[2]):
        DoG = DoG_pyramid[:,:,i]
        curvature = principal_curvature[:,:,i]
        (coords, Dmax) = get_level_extremas(DoG, curvature, th_contrast, th_r) 

        for (row, col) in coords:
            neighbors = []
            for j in [i-1, i+1]:
                if(0 <= j and j < DoG_pyramid.shape[2]):
                    neighbors.append(DoG_pyramid[row, col, j])
            if is_patch_extrema(neighbors, DoG, row, col, Dmax):
                loc = (col-1, row-1, DoG_levels[i])
                locsDoG.append(loc)

    locsDoG = np.stack(locsDoG, axis=0)
    return locsDoG

def get_level_extremas(DoG, curvature, th_c, th_r):
    DoG_max = ndimage.maximum_filter(DoG, size=3, mode='constant')
    DoG_min = ndimage.minimum_filter(DoG, size=3, mode='constant')

    extrema_mask = (DoG_max == DoG) | (DoG_min == DoG) 
    extrema_mask = extrema_mask & ((DoG > th_c) | (DoG < -th_c)) & (curvature < (th_r+1)*(th_r+1)/(th_r))

    coords = np.transpose(np.nonzero(extrema_mask))
    return coords, DoG_max == DoG
    
    
def is_patch_extrema(neighbors, im, row, col, Dmax):
    this = im[row, col]
    maximum = Dmax[row, col]
    for neighbor in neighbors:
        if(maximum and this < neighbor):
            return False
        elif (this > neighbor):
            return False

    return True

def DoGdetector(im, sigma0=1, k=np.sqrt(2), levels=[-1,0,1,2,3,4], 
                th_contrast=0.03, th_r=12):
    '''
    Putting it all together

    Inputs          Description
    --------------------------------------------------------------------------
    im              Grayscale image with range [0,1].

    sigma0          Scale of the 0th image pyramid.

    k               Pyramid Factor.  Suggest sqrt(2).

    levels          Levels of pyramid to construct. Suggest -1:4.

    th_contrast     DoG contrast threshold.  Suggest 0.03.

    th_r            Principal Ratio threshold.  Suggest 12.

    Outputs         Description
    --------------------------------------------------------------------------

    locsDoG         N x 3 matrix where the DoG pyramid achieves a local extrema
                    in both scale and space, and satisfies the two thresholds.

    gauss_pyramid   A matrix of grayscale images of size (imH,imW,len(levels))
    '''
    ##########################
    # TO DO ....
    # compupte gauss_pyramid, gauss_pyramid here
    gauss_pyramid = createGaussianPyramid(im, sigma0=sigma0, k=k, levels=levels)
    DoG_pyr, DoG_levels = createDoGPyramid(gauss_pyramid, levels)
    pc_curvature = computePrincipalCurvature(DoG_pyr)
    locsDoG = getLocalExtrema(DoG_pyr, DoG_levels, pc_curvature, th_contrast, th_r)
    return locsDoG, gauss_pyramid

if __name__ == '__main__':
    # test gaussian pyramid
    levels = [-1,0,1,2,3,4]
    im = cv2.imread('../data/model_chickenbroth.jpg')
    #im = cv2.imread('../data/chickenbroth_01.jpg')
    # im = cv2.imread('../data/incline_L.png')
    #im = cv2.imread('../data/pf_floor.jpg')

    # im_pyr = createGaussianPyramid(im)
    # displayPyramid(im_pyr)
    # # test DoG pyramid
    # DoG_pyr, DoG_levels = createDoGPyramid(im_pyr, levels)
    # displayPyramid(DoG_pyr)
    # # test compute principal curvature
    # pc_curvature = computePrincipalCurvature(DoG_pyr)
    # displayPyramid(pc_curvature)
    # # test get local extrema
    # th_contrast = 0.03
    # th_r = 12
    #locsDoG = getLocalExtrema(DoG_pyr, DoG_levels, pc_curvature, th_contrast, th_r)
    # test DoG detector
    #locsDoG, gaussian_pyramid = DoGdetector(im)
    k = np.sqrt(2)
    levels=[-1,0,1,2,3,4]
    locsDoG, gaussian_pyramid = DoGdetector(im, k=k, levels=levels)
    displayKeypoints(locsDoG, gaussian_pyramid)



