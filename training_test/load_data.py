import numpy as np


def loadMNIST(prefix, folder):
    intType = np.dtype('int32').newbyteorder('>')
    nMetaDataBytes = 4 * intType.itemsize

    data = np.fromfile(folder + "/" + prefix + '-images.idx3-ubyte', dtype='ubyte')
    magicBytes, nImages, width, height = np.frombuffer(data[:nMetaDataBytes].tobytes(), intType)
    data = data[nMetaDataBytes:].astype(dtype='float32').reshape([nImages, width * height])

    labels = np.fromfile(folder + "/" + prefix + '-labels.idx1-ubyte',
                         dtype='ubyte')[2 * intType.itemsize:]

    return data.T/255., labels


trainingImages, trainingLabels = loadMNIST("train", "../data")
testImages, testLabels = loadMNIST("t10k", "../data")

def toHotEncoding(classification):
    # emulates the functionality of tf.keras.utils.to_categorical( y )
    hotEncoding = np.zeros([len(classification),
                            np.max(classification) + 1])
    hotEncoding[np.arange(len(hotEncoding)), classification] = 1
    return hotEncoding.T


trainingLabels = toHotEncoding(trainingLabels)
testLabels = toHotEncoding(testLabels)

print(f"Size of training images: {trainingImages.shape}")
print(f"Size of training labels: {trainingLabels.shape}")



