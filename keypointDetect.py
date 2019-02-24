import numpy as np
import cv2
from scipy import ndimage
from skimage.feature import peak_local_max

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

def displayKeypoints(locs, im_pyr):
    im_pyr = np.split(im_pyr, im_pyr.shape[2], axis=2)
    im = im_pyr[0]
    locs = np.split(locs, locs.shape[1], axis=1)
    im = np.repeat(im.reshape((im.shape[0], im.shape[1], 1)), 3, axis=2)
    for l in range(len(locs)):
        loc = locs[l]
        row = loc[0]
        col = loc[1]
        cv2.circle(im, (col-1, row-1), 1, (0,1,0), thickness=1, lineType=8)
    im = cv2.normalize(im, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.imshow('Pyramid of image', im)
    cv2.waitKey(0) # press any key to exit
    cv2.destroyAllWindows()

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
        Dxx = cv2.Sobel(DoG,-1, 2, 0)
        Dyy = cv2.Sobel(DoG,-1, 0, 2)
        Dxy = cv2.Sobel(DoG,-1, 1, 1)
        Tr = Dxx + Dyy 
        Det = cv2.multiply(Dxx,Dyy) - cv2.multiply(Dxy,Dxy)
        Det[Det==0] = 10000
        R = cv2.divide(cv2.multiply(Tr, Tr), Det)
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
    for i in range(1, DoG_pyramid.shape[2]-1):
        DoG = DoG_pyramid[:,:,i].reshape(DoG_pyramid.shape[0:2])
        curvature = principal_curvature[:,:,i].reshape(principal_curvature.shape[0:2])
        (coords, Dmax, Dmin) = get_level_extremas(DoG, curvature, th_contrast, th_r) 

        valid = []
        for j in [i-1, i+1]:
            other = DoG_pyramid[:,:,j].reshape(DoG_pyramid.shape[0:2])
            for (row, col) in coords:
                if (row >= 1) and (row < im.shape[0]) and (col > 1) and (col < im.shape[1]):
                    (pmax, pmin) = extract_patch_extrema(other, row, col)
                    if (Dmax[row, col] and DoG[row,col] > pmax) or (Dmin[row, col] and DoG[row, col] < pmin):
                        if((row, col, DoG_levels[i]) in valid):
                            locsDoG.append((row, col, DoG_levels[i]))
                        else:
                            valid.append((row, col, DoG_levels[i]))
    locsDoG = np.stack(locsDoG, axis=1)
    return locsDoG

def get_level_extremas(DoG, curvature, th_c, th_r):
    DoG_max = ndimage.maximum_filter(DoG, size=3, mode='constant')
    DoG_min = ndimage.minimum_filter(DoG, size=3, mode='constant')

    extrema_mask = (DoG_max == DoG) | (DoG_min == DoG) 
    extrema_mask = extrema_mask & ((DoG > th_c) | (DoG < -th_c))
    # extrema_mask = peak_local_max(DoG, min_distance=1, indices=False)
    # extrema_mask = extrema_mask & (DoG > th_c) 
    mask_r = curvature > th_r
    extrema_mask = extrema_mask & mask_r
    coords = np.transpose(np.nonzero(extrema_mask))

    return coords, DoG_max == DoG, DoG_min == DoG
    
    
def extract_patch_extrema(im, row, col):
    patch = im[row-1:row+1, col-1:col+1]
    pmin = np.amin(patch)
    pmax = np.amax(patch)
    return (pmax, pmin)

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
    #im = cv2.imread('../data/incline_L.png')
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
    locsDoG, gaussian_pyramid = DoGdetector(im)
    displayKeypoints(locsDoG, gaussian_pyramid)


