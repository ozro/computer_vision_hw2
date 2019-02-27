import numpy as np
import cv2
from BRIEF import briefLite, briefMatch
import matplotlib.pyplot as plt

def computeH(p1, p2):
    '''
    INPUTS:
        p1 and p2 - Each are size (2 x N) matrices of corresponding (x, y)'  
                 coordinates between two images
    OUTPUTS:
     H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear 
            equation
    '''
    assert(p1.shape[1]==p2.shape[1])
    assert(p1.shape[0]==2)
    #############################
    # TO DO ...
    u = p1[0, :].T
    v = p1[1, :].T
    x = p2[0, :].T
    y = p2[1, :].T
    A = constructA(u,v,x,y)
    h = getEigenvector(np.matmul(A.T,A))
    H2to1 = np.reshape(h, (3,3))
    return H2to1

def constructA(u,v,x,y):
    xu = x * u
    xv = x * v
    yu = y * u
    yv = y * v
    one = np.ones(u.shape[0])
    zero = np.zeros(u.shape[0])
    A1 = np.column_stack((-u, -v, -one, zero, zero, zero , xu, xv, x))
    A2 = np.column_stack((zero, zero, zero, -u, -v, -one , yu, yv, y))
    A = np.hstack([A1, A2]).reshape(A1.shape[0] + A2.shape[0], A1.shape[1])
    return A

def getEigenvector(A):
    _,_,V = np.linalg.svd(A)
    h = V[-1, :]
    return h


def ransacH(matches, locs1, locs2, num_iter=5000, tol=2):
    '''
    Returns the best homography by computing the best set of matches using
    RANSAC
    INPUTS
        locs1 and locs2 - matrices specifying point locations in each of the images
        matches - matrix specifying matches between these two sets of point locations
        nIter - number of iterations to run RANSAC
        tol - tolerance value for considering a point to be an inlier

    OUTPUTS
        bestH - homography matrix with the most inliers found during RANSAC
    ''' 
    ###########################
    # TO DO ...
    hlocs1 = locs1[matches[:, 0], :].T
    hlocs1[2,:] = np.ones((1, hlocs1.shape[1]))
    hlocs2 = locs2[matches[:, 1], :].T
    hlocs2[2,:] = np.ones((1, hlocs2.shape[1]))

    bestScore = 0
    bestSSD = 0
    bestInliers = 0
    i = 0
    while(i < num_iter):
        # Pick four correspondences randomly
        idx = np.random.choice(hlocs1.shape[1], 4)
        pt1 = hlocs1[0:2, idx] 
        pt2 = hlocs2[0:2, idx]

        H2to1 = computeH(pt1, pt2)
        H1to2 = computeH(pt2, pt1)

        # Check SSD
        SSD = getSSD(hlocs1, hlocs2, H2to1, H1to2)
        inliers = SSD < tol
        score = np.count_nonzero(inliers)
        SSD_sum = np.sum(SSD)

        # Update best
        if score > bestScore or (score == bestScore and SSD_sum < bestSSD):
            bestSSD = SSD_sum
            bestScore = score
            bestInliers = inliers
            print("\nUpdated!")
            print(score)
            print(score/matches.shape[0])
            print(SSD_sum/score)
            print(hlocs1[0:2, idx])
            print(getTransform(hlocs1[:, idx], H2to1)[0:2,:])
        
        i+=1

    pt1 = hlocs1[0:2, bestInliers] 
    pt2 = hlocs2[0:2, bestInliers]
    print(pt1)
    print(pt2)
    bestH = computeH(pt1, pt2)
    return bestH

def getSSD(hlocs1, hlocs2, H2to1, H1to2):
    transformed_hlocs2 = getTransform(hlocs2, H2to1) 
    transformed_hlocs1 = getTransform(hlocs1, H1to2) 
    ssd1 = np.sum((hlocs1[0:2,:]-transformed_hlocs2[0:2,:])**2, axis=0)
    ssd2 = np.sum((hlocs2[0:2,:]-transformed_hlocs1[0:2,:])**2, axis=0)
    return ssd1+ssd2

def getTransform(pts, H):
    transformed = np.matmul(H, pts)
    scale = np.tile(transformed[3,:], (3,1))
    transformed = transformed/scale
    return transformed

def displaySide(im1, im2):
    fig = plt.figure()
    # draw two images side by side
    imH = max(im1.shape[0], im2.shape[0])
    im = np.zeros((imH, im1.shape[1]+im2.shape[1]), dtype='uint8')
    im[0:im1.shape[0], 0:im1.shape[1]] = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im[0:im2.shape[0], im1.shape[1]:] = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    plt.imshow(im, cmap='gray')
    plt.show()

if __name__ == '__main__':
    # np.set_printoptions(precision=3)
    # p1 = np.array([[0, 1, 1, 0], [0, 1, 0, 1], [1, 1, 1, 1]])
    # p2 = p1.copy()
    # H = computeH(p1[0:2, :], p2[0:2, :])

    # # Test RANSAC H
    im1 = cv2.imread('../data/model_chickenbroth.jpg')
    im2 = cv2.imread('../data/model_chickenbroth.jpg')
    #im2 = cv2.imread('../data/chickenbroth_01.jpg')
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    bestH = ransacH(matches, locs1, locs2, num_iter=5000, tol=4)
    print(bestH)

    # # Compare warped image with source
    im_warped = cv2.warpPerspective(im2, bestH, (im2.shape[1], im2.shape[0]))
    displaySide(im1, im_warped)
