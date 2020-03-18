import sys
sys.path.append('cmake-build-release')
import numpy as np
import gadgetron_toolbox
from scipy.misc import face
import matplotlib.pyplot as plt
import skimage.transform as st
from scipy.misc import imread

if __name__ == '__main__':

    # image = face(True).astype(np.float32)
    image1 = imread('lenag1.png',flatten=True)
    image2 = imread('lenag2.png',flatten=True)

    vfield = gadgetron_toolbox.demons(image1,image2, 200,2.0)

    registered_image = gadgetron_toolbox.deform_image(image1,vfield)

    print(vfield)

    plt.imshow(registered_image,cmap='gray')
    plt.figure()
    plt.imshow(vfield[:,:,0])
    plt.figure()
    plt.imshow(vfield[:,:,1])
    plt.show()

