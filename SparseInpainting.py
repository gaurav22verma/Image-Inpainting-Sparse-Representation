import scipy
import numpy as np
from sklearn.feature_extraction import image
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import pickle

MAX_PATCHES = 256
PATCH_SIZE = (8, 8)
IMAGE_SIZE = (512, 512)
N_NONZERO_COEFFS = 7
STRIDE = 3
MAX_ITERS_IRLS = 100
IRLS_TOLERANCE = 0.001
rng = np.random.RandomState(42)

def IterativeReweightedLeastSquares(dictionary, y, K = N_NONZERO_COEFFS):
    # initialize sparseCoeff and W
    sparseCoeff = np.ones((MAX_PATCHES, 1))
    W = np.identity(MAX_PATCHES)
    for _ in range(MAX_ITERS_IRLS):
        _sparseCoeff = sparseCoeff
        sparseCoeff = np.matmul(np.matmul(np.matmul(np.matmul(W, W), dictionary.T),\
                        np.linalg.inv(np.matmul(dictionary, np.matmul((np.matmul(W, W)),\
                         dictionary.T)))), y)
        W = np.diag(sparseCoeff.reshape((sparseCoeff.shape[0]),))
        tolerance = sum(abs(sparseCoeff - _sparseCoeff))
        if tolerance <= IRLS_TOLERANCE:
            return sparseCoeff
    sparseCoeff = sparseCoeff.astype(np.float32)
    #sparseCoeff = np.abs(sparseCoeff)
    return sparseCoeff

def OrthogonalMatchingPursuit(dictionary, y, K = N_NONZERO_COEFFS):
    residual = y
    support = []
    for var in range(K):
        product = np.matmul(dictionary.T, residual)
        maxElt = int(product.argmax(axis = 0))
        support.append(maxElt)
        supportDictionary = dictionary[:, support]
        xfinal, resid, rank, s = np.linalg.lstsq(supportDictionary, y)
        residual = y - np.matmul(supportDictionary, xfinal)
    xfinalSparse = np.zeros((MAX_PATCHES, 1))
    for indexNo in range(MAX_PATCHES):
        if indexNo in support:
            xfinalSparse[indexNo] = float(xfinal[support.index(indexNo)])
    xfinalSparse = xfinalSparse.astype(np.float32)
    #xfinalSparse = np.abs(xfinalSparse)
    return xfinalSparse

# Shows the dictionary
def ShowDictionary(dictPatches, figsize = (4.2, 4)):
    for i, patch in enumerate(dictPatches):
        plt.subplot(16, 16, i + 1)
        plt.imshow(patch.reshape(8, 8), cmap = plt.cm.gray, interpolation = 'bilinear')
        plt.xticks(())
        plt.yticks(())
    plt.suptitle('Dictionary | (64, 256)')
    plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
    plt.show()
    return

# Returns a (MAX_PATCHES, NUM_PIXELS) dimensional matrix of randomly extracted patches from input image
def GetVectorizedDictionary(imageFile, max_patches = 256, patch_size = (8, 8)):
    imageFile = imageFile/255.0
    patches = image.extract_patches_2d(imageFile, patch_size, max_patches = max_patches, random_state = rng)
    patches = patches.reshape((patches.shape[0], -1))
    patches -= np.mean(patches, axis = 0)
    patches /= np.std(patches, axis=0)
    return patches

filenames = ['aditya_face_crop.jpg', 'daniel_face_crop.jpg', 'keanu_face_crop.jpg',\
            'natalie_face_crop.jpg', 'vikulp_face_crop.jpg', 'harry.jpg']
testImage = 'harry.jpg'
maskedImage = 'harry_mask_only.jpg'
baseDir = 'images/'

# Get 256 patches from all the input images
allPatches = []
for a_filename in filenames:
    imageName = baseDir + a_filename
    imageFile = scipy.misc.imread(imageName, mode = 'I')
    imageFile = scipy.misc.imresize(imageFile, IMAGE_SIZE, interp='bilinear')
    # plt.imshow(imageFile, cmap = plt.cm.gray, interpolation = 'nearest')
    # plt.show()
    a_patchSet = GetVectorizedDictionary(imageFile, MAX_PATCHES, PATCH_SIZE)
    allPatches.append(a_patchSet)
allPatches = [item for sublist in allPatches for item in sublist]
allPatches = np.array(allPatches)


# Out of allPatches, randomly choose MAX_PATCHES patches and call them dictionary
randomIndices = np.random.randint(0, allPatches.shape[0], size = (MAX_PATCHES,))
dictPatches = [allPatches[var] for var in randomIndices]
dictPatches = np.array(dictPatches)
#ShowDictionary(dictPatches)


# Normalize the columns of the dictionary, after taking transpose
dictPatches = dictPatches.astype(np.float32)
dictPatches = dictPatches.T
#dictPatches = dictPatches/np.linalg.norm(dictPatches, axis = 0)

# Input image: The original image
imageFile = scipy.misc.imread(baseDir + testImage, mode = 'I')
imageFile = scipy.misc.imresize(imageFile, IMAGE_SIZE, interp='bilinear')
imageFile = np.array(imageFile).astype(np.float32)/255.0

tempImageFile = np.copy(imageFile)

# Read the masked image from .pkl file
with open('currentMaskHarry.pkl') as f:
    maskedImage = pickle.load(f)
    maskedImage = np.array(maskedImage)
    maskedImage = maskedImage.reshape(IMAGE_SIZE)

# This is the variable in which our reconstructed image will be stored
imageReconstruction = np.zeros((IMAGE_SIZE))
imageReconstruction = imageReconstruction.astype(np.float32)

# First copy paste the parts of the image that are not to be inpainted
for rowJump in range(0, IMAGE_SIZE[0] - 7, STRIDE):
    for colJump in range(0, IMAGE_SIZE[1] - 7, STRIDE):
        y = maskedImage[rowJump:rowJump + 8, colJump: colJump + 8]
        y = y.reshape(1, PATCH_SIZE[0]*PATCH_SIZE[1])
        y = y.T
        if 1.0 not in y:
            imageReconstruction[rowJump:rowJump + 8, colJump:colJump + 8] = imageFile[rowJump:rowJump + 8, colJump:colJump + 8]
        else:
            continue

# Figure out the average pixel value
avgPixelValue = np.mean(imageFile)

# Do this till the maskedImage has no 1.0 in it
count = 0
while True:
    count += 1
    print count
    if maskedImage.any() == 1:
        # Get the patch that has least number of non-zero missing pixels
        missingPixels = []
        coordinates = []
        for rowJump in range(0, IMAGE_SIZE[0]-7, STRIDE):
            for colJump in range(0, IMAGE_SIZE[1]-7, STRIDE):
                y = maskedImage[rowJump:rowJump + 8, colJump:colJump + 8]
                y = y.reshape((1, PATCH_SIZE[0]*PATCH_SIZE[1]))
                y = y.T # Here y is a (64, 1) dimensional vector
                if 1.0 in y:
                    unique, counts = np.unique(y, return_counts = True)
                    someVar = dict(zip(unique, counts))
                    missingPixels.append(someVar[1.0])
                    coordinates.append([rowJump, colJump])
                else:
                    continue
        # Determine the patch that has least number of missing pixels
        patchLocation = coordinates[missingPixels.index(min(missingPixels))]
        rowNo = patchLocation[0]
        colNo = patchLocation[1]
        # This is where the reconstruction begins
        y = tempImageFile[rowNo:rowNo + 8, colNo:colNo + 8]
        y = y.reshape(1, PATCH_SIZE[0]*PATCH_SIZE[1])
        y = y.T
        refMask = maskedImage[rowNo:rowNo + 8, colNo:colNo + 8]
        refMask = refMask.reshape(1, PATCH_SIZE[0]*PATCH_SIZE[1])
        refMask = refMask.T
        maskedImage[rowNo:rowNo + 8, colNo:colNo + 8] = 0.0
        # Remove the rows of y and dictPatches that correspond to 1.0 in refMask
        delRows = []
        for someVar in range(refMask.shape[0]):
            if refMask[someVar] == 1:
                delRows.append(someVar)
        ysub = np.delete(y, delRows, 0)
        dictPatchesSub = np.delete(dictPatches, delRows, 0)
        # What if ysub is empty?
        if ysub.shape[0] == 0:
            yhat = np.ones((PATCH_SIZE[0]*PATCH_SIZE[1]))*avgPixelValue
            yhat = yhat.astype(np.float32)
            imageReconstruction[rowNo:rowNo + 8, colNo:colNo + 8] = yhat.reshape(8, 8)
            continue
        # Calculate the coef using ysub and dictPatchesSub
        coef = OrthogonalMatchingPursuit(dictPatchesSub, ysub, N_NONZERO_COEFFS)
        #coef = IterativeReweightedLeastSquares(dictPatchesSub, ysub, N_NONZERO_COEFFS)
        yhat = np.matmul(dictPatches, coef)
        yhat = yhat.astype(np.float32)
        if max(yhat) >= 1.0 or min(yhat) <= 0.0:
            yhat = np.ones((PATCH_SIZE[0]*PATCH_SIZE[1]))*avgPixelValue
            yhat = yhat.astype(np.float32)
            imageReconstruction[rowNo:rowNo + 8, colNo:colNo + 8] = yhat.reshape(8, 8)
            continue
        imageReconstruction[rowNo:rowNo + 8, colNo:colNo + 8] = yhat.reshape(8, 8)
        tempImageFile[rowNo:rowNo + 8, colNo:colNo + 8] = yhat.reshape(8, 8)
    else:
        break

scipy.misc.imsave('OUTPUT_HARRY_NEW.jpg', imageReconstruction)

plt.subplot(1, 2, 1)
plt.imshow(imageFile, vmin = 0, vmax = 1, cmap = plt.cm.gray, interpolation = 'bilinear')
plt.xticks(())
plt.yticks(())
plt.subplot(1, 2, 2)
plt.imshow(imageReconstruction, vmin = 0, vmax = 1, cmap = plt.cm.gray, interpolation = 'bilinear')
plt.xticks(())
plt.yticks(())
plt.show()
