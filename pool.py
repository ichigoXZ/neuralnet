import numpy as np
from scipy.misc import imread, imsave

def maxpool(input, kernel_size, stride):
    batch, in_weight, in_height, in_channels = input.shape
    out_weight = (in_weight - kernel_size[0] + 1) / stride[0]
    out_height = (in_height - kernel_size[1] + 1) / stride[1]
    output = np.zeros([batch, out_weight, out_height, in_channels])
    for b in range(batch):
        for i in range(out_weight):
            for j in range(out_height):
                for c in range(in_channels):
                    output[b, i, j, c] = np.max(
                        input[b, i*stride[0]:(i+1)*stride[0]-1,
                                j*stride[1]:(j+1)*stride[1]-1, c]
                    )
    return output

def avgpool(input, kernel_size, stride):
    batch, in_weight, in_height, in_channels = input.shape
    out_weight = (in_weight - kernel_size[0] + 1) / stride[0]
    out_height = (in_height - kernel_size[1] + 1) / stride[1]
    output = np.zeros([batch, out_weight, out_height, in_channels])
    for b in range(batch):
        for i in range(out_weight):
            for j in range(out_height):
                for c in range(in_channels):
                    output[b, i, j, c] = np.average(
                        input[b, i*stride[0]:(i+1)*stride[0]-1,
                                j*stride[1]:(j+1)*stride[1]-1, c]
                    )
    return output

if __name__ == "__main__":
    image = imread("image/tree.jpg")
    image = np.resize(image,[1,image.shape[0],image.shape[1],image.shape[2]])
    print "image.shape",image.shape
    output = avgpool(image, [2,2],[2,2])
    print output.shape
    output = np.resize(output, [output.shape[1], output.shape[2], output.shape[3]])
    # print output
    imsave("image/tree_output_pool.jpg",output)

