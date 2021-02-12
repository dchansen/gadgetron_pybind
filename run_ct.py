import sys
sys.path.append('build')
import numpy as np
import gadgetron_toolbox
from scipy.misc import face
import matplotlib.pyplot as plt
import skimage.transform as st
from scipy.ndimage import gaussian_filter1d
from imageio import imread
import imageio


def demons_step(fixed, moving, alpha=0.5, beta=1e-6):
    dfixed = np.stack(np.gradient(fixed))
    dmoving = np.stack(np.gradient(moving))
    daverage = (dfixed + dmoving) / 2

    it = moving - fixed

    result = it * daverage / (np.sum(daverage ** 2, axis=0) + (alpha * it) ** 2 + beta)
    result = np.transpose(result,[1,2,0])
    result = np.roll(result,1,axis=2)
    return result


if __name__ == '__main__':

    image1 = imageio.imread('lenag1.png').astype(np.float32)/256
    # image = 255-image
    image2 = imageio.imread('lenag2.png').astype(np.float32)/256


    print(np.linalg.norm(image1-image2))
    #vfield = gadgetron_toolbox.demons(image1,image2,500,2.0)
    #vfield = gadgetron_toolbox.multi_ngf_demons(image1,image2,3,500,2.0)

    vfield = gadgetron_toolbox.cmr_registration(image2[np.newaxis],image1[np.newaxis])
    vfield = vfield[0,:,:]
    print(vfield.shape)


    registered_image = gadgetron_toolbox.deform_image_bspline(image1,vfield)
    print(registered_image.shape)
    print(np.linalg.norm(registered_image-image2))
    #print(vfield.shape)
    plt.figure()
    plt.imshow(image1,cmap='gray')
    plt.figure()
    plt.imshow(image2,cmap='gray')
    plt.figure()
    plt.imshow(registered_image,cmap='gray')
    # plt.figure()
    # plt.imshow(vfield[:,:,0])
    # plt.figure()
    # plt.imshow(vfield[:,:,1])
    plt.show()

