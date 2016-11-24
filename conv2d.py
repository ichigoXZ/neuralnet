from scipy.misc import imread, imsave
import numpy as np

filters = {"edges":np.array([[0, 0, 0, 0, 0],
                    [0, 0, -2, 0, 0],
                    [0, -2, 8, -2, 0],
                    [0, 0, -2, 0, 0],
                    [0, 0, 0, 0, 0]])}

def conv2d(input, filters, filter_shape, strides=1, padding="valid"):
    batch, in_weight, in_height, in_channel = input.shape
    _, filter_weight, filter_height, out_channel = filter_shape
    if input.shape[3] != filter_shape[0]:
        raise "input in_channels and filter in_channels not fit!"
    output = np.zeros([batch, in_weight-filter_weight+1, in_height-filter_height+1, out_channel])
    for b in range(batch):
        for c in range(out_channel):
            print "channel:", c
            for i in range(in_weight-filter_weight+1):
                for j in range(in_height-filter_height+1):
                    x = np.sum(np.dot(input[b, i:i+filter_weight, j:j+filter_height, c], filters))
                    if x>0:
                        output[b, i, j, c] = x
    return output


if __name__ == "__main__":
    image = imread("image/tree.jpg")
    image = np.resize(image,[1,image.shape[0],image.shape[1],image.shape[2]])
    print "image.shape",image.shape
    output = conv2d(image, filters=filters['edges'],filter_shape=[3,5,5,3])
    print output.shape
    output = np.resize(output, [output.shape[1], output.shape[2], output.shape[3]])
    # print output
    imsave("image/tree_output.jpg",output)