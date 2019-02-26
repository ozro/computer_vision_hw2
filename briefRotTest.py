import BRIEF
import numpy as np
import cv2
import matplotlib.pyplot as plt

im1 = cv2.imread('../data/model_chickenbroth.jpg')
results = []
for ang in range(0, 360, 10):
    rot = cv2.getRotationMatrix2D((im1.shape[1]/2, im1.shape[0]/2), ang, 1)
    im2 = cv2.warpAffine(im1, rot, (im1.shape[1], im1.shape[0]))
    locs1, desc1 = BRIEF.briefLite(im1)
    locs2, desc2 = BRIEF.briefLite(im2)
    matches = BRIEF.briefMatch(desc1, desc2)
    results.append(len(matches))

objects = [i for i in range(0, 360, 50)]
ticks = np.arange(len(objects)) * 5
y_pos = np.arange(360/10)
 
plt.bar(y_pos, results, align='center', alpha=0.5)
plt.xticks(ticks, objects)
plt.ylabel('Number of matches')
plt.xlabel('Rotation (degrees)')
plt.title('BRIEF Matches vs Rotation')
 
plt.show()