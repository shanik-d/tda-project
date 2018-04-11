import matplotlib.pyplot as plt
import cv2

fpath = "../images/sunsets/sunset ("
imArr=[]
grArr=[]
for i in range(13):
    num = str(i+1)
    filepath = fpath + num + ").jpg"
    image = cv2.imread(filepath)
    imArr.append(image)
    grArr.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    #to use the SIFT features we need a greyscale image

# we can dispaly it with matplotlib

def show_colour_and_grey_image(img_index):
    colour_img = cv2.cvtColor(imArr[img_index], cv2.COLOR_BGR2RGB)
    grey_img = grArr[img_index]
    fig = plt.figure()
    ax0 = fig.add_subplot(121)
    ax0.axis("off")
    ax0.imshow(colour_img)
    ax1 = fig.add_subplot(122)
    ax1.axis("off")
    ax1.imshow(grey_img, cmap='gray')
    plt.show()

show_colour_and_grey_image(1)
show_colour_and_grey_image(8)

def gen_sift_features(gray_img):
    sift = cv2.xfeatures2d.SIFT_create()
    # kp is the keypoints
    #
    # desc is the SIFT descriptors, they're 128-dimensional vectors
    # that we can use for our final features
    kp, desc = sift.detectAndCompute(gray_img, None)
    return kp, desc

def show_sift_features(gray_img, color_img, kp):
    plt.figure()
    plt.axis("off")
    plt.imshow(cv2.drawKeypoints(gray_img, kp, color_img.copy()))
    plt.show()

# first_image = imArr[0]
# first_image_grey = grArr[0]
# first_image_kp, first_image_desc = gen_sift_features(first_image_grey)
#
# print ('Here are what our SIFT features look like for the front-view octopus image:')
# show_sift_features(first_image_grey, first_image, first_image_kp)

sunset_1 = imArr[1]
sunset_1_grey = grArr[1]
sunset_1_kp, sunset_1_desc = gen_sift_features(sunset_1_grey)
show_sift_features(sunset_1_grey, sunset_1, sunset_1_kp)

sunset_2 = imArr[8]
sunset_2_grey = grArr[8]
sunset_2_kp, sunset_2_desc = gen_sift_features(sunset_2_grey)
show_sift_features(sunset_2_grey, sunset_2, sunset_2_kp)
