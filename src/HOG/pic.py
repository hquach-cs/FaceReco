# Histogram of Oriented Gradients
import matplotlib.pyplot as plt
from skimage import data, color, feature
import skimage.data
import cv2
import sys

# Ask for path to image
if(len(sys.argv) < 2):
    print("Enter One Picture Path for Face Detection.")
    exit(0)

# Reads image basd on preference
image = cv2.cvtColor(cv2.imread(sys.argv[1]), cv2.COLOR_BGR2RGB)
# image = color.rgb2gray(cv2.imread(sys.argv[1]))
hog_vec, hog_vis = feature.hog(image, visualize=True)

# Setup axis for imagery
fig, ax = plt.subplots(1, 2, figsize=(12, 6),
                    subplot_kw=dict(xticks=[], yticks=[])) #To not show grid axis

# Setup imagery 
ax[0].imshow(image)
ax[0].set_title('input image')

ax[1].imshow(hog_vis)
ax[1].set_title('visualization of HOG features')
plt.show()
