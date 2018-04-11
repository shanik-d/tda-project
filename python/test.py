# import the necessary packages
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
#import argparse
#import utils
import cv2

fpath = "../images/sunsets/sunset ("
imArr=[]
for i in range(13):
    num = str(i+1)
    filepath = fpath + num + ").jpg"
    imArr.append(cv2.imread(filepath))
    imArr[i] = cv2.cvtColor(imArr[i], cv2.COLOR_BGR2RGB)
# load the image and convert it from BGR to RGB so that
# we can dispaly it with matplotlib

image = imArr[0]

# show our image
plt.figure()
plt.axis("off")
plt.imshow(image)
plt.show()


# reshape the image to be a list of pixels
image = image.reshape((image.shape[0] * image.shape[1], 3))

clt = KMeans(n_clusters = 5)
clt.fit(image)
