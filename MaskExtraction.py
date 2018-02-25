import scipy.misc
import numpy as np
import pickle
import matplotlib.pyplot as plt

IMAGE_SIZE = (512, 512)

baseDir = 'images/'
inputImageMasked = 'harry_mask.jpg'
inputImageOrg = 'harry.jpg'

imageFileMasked = scipy.misc.imread(baseDir + inputImageMasked, mode = 'I')
imageFileMasked = scipy.misc.imresize(imageFileMasked, IMAGE_SIZE, interp='bilinear')
imageFileMasked = np.array(imageFileMasked).astype(np.float32)/255.0

imageFileOrg = scipy.misc.imread(baseDir + inputImageOrg, mode = 'I')
imageFileOrg = scipy.misc.imresize(imageFileOrg, IMAGE_SIZE, interp='bilinear')
imageFileOrg = np.array(imageFileOrg).astype(np.float32)/255.0

mask = np.zeros(IMAGE_SIZE)
for rowJump in range(IMAGE_SIZE[0]):
    for colJump in range(IMAGE_SIZE[1]):
        if imageFileMasked[rowJump, colJump] >= 0.94:
            mask[rowJump, colJump] = 1.0
        else:
            continue

plt.imshow(mask, vmin = 0, vmax = 1, cmap = plt.cm.gray, interpolation = 'bilinear')
plt.show()
scipy.misc.imsave('images/harry_mask_only.jpg', mask)
with open('currentMaskHarry.pkl', 'w') as f:
    pickle.dump([mask], f)
