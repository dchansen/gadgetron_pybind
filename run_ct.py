import sys
sys.path.append('cmake-build-release')
import numpy as np
import gadgetron_toolbox
from scipy.misc import face
import matplotlib.pyplot as plt
import skimage.transform as st
from scipy.misc import imread
import imageio
if __name__ == '__main__':

    image1 = imageio.imread('lenag1.png').astype(np.float32)/256
    # image = 255-image
    image2 = imageio.imread('lenag2.png').astype(np.float32)/256

    print(np.linalg.norm(image1-image2))
    vfield = gadgetron_toolbox.demons(image1,image2, 200,2.0)

    registered_image = gadgetron_toolbox.deform_image(image1,vfield)
    print(np.linalg.norm(registered_image-image2))
    print(vfield.shape)

    plt.imshow(registered_image,cmap='gray')
    plt.figure()
    plt.imshow(vfield[:,:,0])
    plt.figure()
    plt.imshow(vfield[:,:,1])
    plt.show()

