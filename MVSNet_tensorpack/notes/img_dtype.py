import cv2
from matplotlib import pyplot as plt

if __name__ == "__main__":
    img_path = '/home/yeeef/Desktop/data/NYUv2pics/nyu_images/0.jpg'
    gray_path = '/home/yeeef/Desktop/data/NYUv2pics/nyu_depths/0.png'

    img = cv2.imread(img_path)
    print(img.dtype, img.shape)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(gray_img.dtype, gray_img.shape)
    gray = cv2.imread(gray_path, cv2.IMREAD_GRAYSCALE)
    print(gray.dtype, gray.shape)

    # plt.figure()
    # plt.imshow((img.astype('float32') / 255.))
    # plt.show()

    plt.figure()
    plt.imshow(gray_img.astype('float32') * 10000)
    plt.show()

    # print(gray_img)


