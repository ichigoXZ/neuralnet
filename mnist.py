# coding:utf-8
import numpy as np
import struct
import matplotlib.pyplot as plt
from scipy.misc import imsave

from dealCsv import writeCsv, readCsv

# imagefile = 'data/train-images.idx3-ubyte'
# labelfile = 'data/train-labels.idx1-ubyte'
imagefile = 'data/t10k-images.idx3-ubyte'
labelfile = 'data/t10k-labels.idx1-ubyte'


def readImages(filename):
    binfile = open(filename, 'rb')
    buf = binfile.read()

    index = 0
    magic, numImages, numRows, numColumns = struct.unpack_from('>IIII', buf, index)
    index += struct.calcsize('>IIII')

    print "magic:", magic
    print "numImages", numImages
    print "numRows:", numRows
    print "numColumns", numColumns

    for i in range(numImages):
        # name = str(i) + ".jpg"
        im = struct.unpack_from('>784B', buf, index)
        index += struct.calcsize('>784B')
        im = np.array(im)
        im = im.reshape(1, 784)
        if i == 0:
            images = im
        else:
            images = np.r_[images, im]

        if i%100 == 0:
            print "image:",i
        # imsave(name, im)

        # fig = plt.figure()
        # im = im.reshape([28,28])
        # plotwindow = fig.add_subplot(111)
        # plt.imshow(im , cmap='gray')
        # plt.show()

    return images

def readLables(filename):
    binfile = open(filename, 'rb')
    buf = binfile.read()

    index = 0
    magic, numLabels = struct.unpack_from('>II', buf, index)
    index += struct.calcsize('>II')

    print magic
    print numLabels

    nums = []
    for i in range(numLabels):
        numtemp = struct.unpack_from('1B', buf, index)
        # numtemp 为tuple类型，读取其数值
        num = numtemp[0]
        index += struct.calcsize('1B')
        nums.append(num)
    nums = np.array(nums)
    return nums


if __name__ == "__main__":
    # images = readImages(imagefile)
    # print images.shape
    labels = readLables(labelfile)
    # writeCsv("data/test_images.csv", images)
    writeCsv("data/test_labels.csv", labels)
    # images = readCsv("data/train_images.csv")
    # print images.shape
    # im = images[3]
    # im = im.reshape([28,28])
    # fig = plt.figure()
    # plotwindow = fig.add_subplot(111)
    # plt.imshow(im , cmap='gray')
    # plt.show()