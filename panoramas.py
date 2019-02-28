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

    im_warped = cv2.warpPerspective(im2, H2to1, (im2.shape[1], im2.shape[0]))
    pano_im = blendImages(im1, im_warped, H2 =H2to1)
    return pano_im

def imageStitching_noClip(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given 
    homography matrix without cliping.
    ''' 
    ######################################
    # TO DO ...
    return pano_im

def blendImages(im1, im2, H1=np.identity(3), H2=np.identity(3)):
    imH = max(im1.shape[0], im2.shape[0])
    imW = max(im1.shape[1], im2.shape[1])
    channels = max(im1.shape[2], im2.shape[2])
    if im1.max()>10:
        im1 = np.float32(im1)/255
    if im2.max()>10:
        im2 = np.float32(im2)/255

    mask1 = generateMask(imW,imH, im1.shape, H1) 
    mask2 = generateMask(imW,imH, im2.shape, H2) 
    mask = mask1 + mask2
    alpha = np.divide(mask1, mask, out=np.ones_like(mask1), where=mask!=0)

    alpha1 = alpha[:im1.shape[0], :im1.shape[1], :]
    alpha2 = 1-(alpha[:im2.shape[0], :im2.shape[1], :])
    cv2.imshow('mask1', mask1)
    cv2.imshow('mask2', mask2)
    cv2.imshow('alpha', alpha)

    im = np.zeros((imH, imW, channels))
    cv2.imshow('im1', im1*alpha1)
    im[:im1.shape[0], :im1.shape[1],:] = im1 * alpha1
    im[:im2.shape[0], :im2.shape[1],:] += im2 * alpha2

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
    print(im1.shape)
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    # plotMatches(im1,im2,matches,locs1,locs2)
    H2to1 = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
    # pano_im = imageStitching_noClip(im1, im2, H2to1)
    pano_im = imageStitching(im1, im2, H2to1)
    print(H2to1)
    cv2.imwrite('../results/panoImg.png', pano_im)
    cv2.imshow('panoramas', pano_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()