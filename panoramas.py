import cv2
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from planarH import ransacH
from BRIEF import briefLite,briefMatch,plotMatches

def imageStitching(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given 
    homography matrix

    INPUT
        Warps img2 into img1 reference frame using the provided warpH() function
        H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear
                 equation
    OUTPUT
        Blends img1 and warped img2 and outputs the panorama image
    '''
    #######################################
    # TO DO ...

    imH = max(im1.shape[0], im2.shape[0])
    imW = max(im1.shape[1], im2.shape[1])
    out_size = (imW, imH)
    pano_im = blendImages(out_size, im1, im2, H2=H2to1)
    return pano_im

def imageStitching_noClip(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given 
    homography matrix without cliping.
    ''' 
    ######################################
    # TO DO ...
    imW = int(im1.shape[1] + im2.shape[1] * 0.5)
    (M, imH) = getTransform(imW, im1.shape, im2.shape, H2to1)
    out_size = (imW, imH)

    pano_im = blendImages(out_size, im1, im2, H1 = M, H2=np.matmul(M, H2to1))
    return pano_im

def getTransform(imW, shape1, shape2, H2to1):
    extrema = np.array([[0, 0], [0, shape2[0]], [shape2[1],0], [shape2[1], shape2[0]]])
    extrema = np.vstack((extrema.T, np.ones((1, extrema.shape[0]))))
    extrema = np.matmul(H2to1, extrema)
    scale = np.tile(extrema[2,:], (3,1))
    extrema = extrema/scale
    minX = np.min(extrema[0,:])
    minY = np.min(extrema[1,:])
    maxX = np.max(extrema[0,:])
    maxY = np.max(extrema[1,:])

    tx = 0
    ty = 0
    scaleX = 1
    if(maxX > imW):
        scaleX = imW / maxX
    if(minX < 0):
        tx = -minX * scaleX
    if(minY < 0):
        ty = -minY * scaleX 
    imH = int(np.ceil((maxY) * scaleX + ty))

    M = np.identity(3)
    M[0,0] = scaleX
    M[1,1] = scaleX
    M[0, 2] = tx
    M[1, 2] = ty
    return (M, imH)

def blendImages(out_size, im1, im2, H1=np.identity(3), H2=np.identity(3)):
    if im1.max()>10:
        im1 = np.float32(im1)/255
    if im2.max()>10:
        im2 = np.float32(im2)/255
    
    imW, imH = out_size

    mask1 = generateMask(imW,imH, im1.shape, H1) 
    mask2 = generateMask(imW,imH, im2.shape, H2) 
    mask = mask1 + mask2
    alpha = np.divide(mask1, mask, out=np.ones_like(mask1), where=mask!=0)

    alpha1 = alpha
    alpha2 = 1-alpha
    # cv2.imshow('mask1', mask1)
    # cv2.imshow('mask2', mask2)
    # cv2.imshow('alpha', alpha)

    im_warped1 = cv2.warpPerspective(im1, H1, out_size)
    im_warped2 = cv2.warpPerspective(im2, H2, out_size)

    im = im_warped1 * alpha1
    im += im_warped2 * alpha2

    return im

def generateMask(imW, imH,shape, H):
    mask = np.zeros(shape)
    mask[0,:] = 1
    mask[-1,:] = 1
    mask[:,0] = 1
    mask[:,-1] = 1
    mask = distance_transform_edt(1-mask)
    mask = mask/mask.max()
    final = cv2.warpPerspective(mask, H, (imW, imH))

    return final

if __name__ == '__main__':
    im1 = cv2.imread('../data/incline_L.png')
    im2 = cv2.imread('../data/incline_R.png')
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    # plotMatches(im1,im2,matches,locs1,locs2)
    H2to1 = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
    # pano_im = imageStitching_noClip(im1, im2, H2to1)
    pano_im = imageStitching(im1, im2, H2to1)
    cv2.imwrite('../results/panoImg.png', pano_im)
    cv2.imshow('panoramas', pano_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()