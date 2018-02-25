import numpy as np
import scipy.misc
import niqe

inputImageName = 'OUTPUT_HARRY.jpg'

inputImage = scipy.misc.imread(inputImageName, flatten = True).astype(np.float32)
print inputImage.shape
niqe = niqe.niqe(inputImage/255.0)

print niqe
