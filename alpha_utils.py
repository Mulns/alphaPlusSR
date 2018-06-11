#!/home/mulns/anaconda3/bin/python
#    or change to /usr/lib/python3
# Created by Mulns at 2018/6/10
# Contact : mulns@outlook.com
# Visit : https://mulns.github.io

import matplotlib.pyplot as plt
from ast import literal_eval
from scipy import ndimage
from scipy import signal
from scipy import misc
from PIL import Image
import numpy as np
import functools
import shutil
import h5py
import sys
import os


def _is_gray(image):
    """Raise ValueError if image is not in gray scale."""
    if len(image.shape) == 2:
        return True
    else:
        raise ValueError(
            "Should be in gray scale. While input has shape of %s" % (str(image.shape)))


def psnr(y_true, y_pred):
    """Calculate the psnr of two images.

        PSNR: Peak Signal to Noise Ratio.
            PSNR = 10 * log(PEAK^2 / MSE) (dB)
        PEAK is the peak value of the signal.
        In this case, we assume that inputs are images so PEAK can only be 1 or 255.
        MSE: Mean Square Error. 
            MSE = (P1 - P2)^2 / N
            P1 and P2 are two images in the same shape, N is the number of pixels.

        Args:
            y_true, y_pred: numpy array in the same shape.

        Returns:
            psnr value of two arrays in numpy.float64

        Raises:
            ValueError: An error occured when two array are in different shape.
    """
    if y_pred.shape != y_true.shape:
        raise ValueError("Input should be in same shape. While %s %s are given." % (
            str(y_pred.shape), str(y_true.shape)))

    def normalize(image):
        if np.max(image) > 1:
            return image/255.
        else:
            return image

    y_pred, y_true = list(map(normalize, [y_pred, y_true]))
    psnr = -10. * np.log10(np.mean(np.square(y_pred - y_true)))
    return psnr.astype("float64")

############################################
# Edge Extraction and Upscaling.
############################################


def innedge(alpha, structure_size=(1, 1), **kargs):
    """Calculate innedge of image.

        Calculate innedge of image, which means the absulute white pixels. It will be a binary image. It uses close/open operation to smooth the edge. We suggest to use open operation to create no error.

        Args:
            alpha: Gray scale image in numpy array, should be normalized.
            structure_size: Size of rectangle used in open/close operation, tuple.
            **kargs: 
                no_smooth: Bool. Set true if using open/close operation to smooth edge.
                mode: "open" or "close", string. Choose to use open or close operation to smooth edge. We use open operation by default.
                outedge: Bool. Set true if you want to generate the outer edge map. You can just call outedge() directly.

        Returns:
            Innedge, a binary image in numpy array.

        Raises:
            ValueError: An error occurred when alpha is not in gray scale.
    """
    _ = _is_gray(alpha)  # if not gray, raise ValueError. See _is_gray().

    if np.max(alpha) > 1:
        alpha /= 255.  # normalize

    if "outedge" in kargs and kargs["outedge"]:
        edge = (alpha != 0)*1  # outedge
    else:
        edge = (alpha == 1)*1  # innedge by default
    if "no smooth" in kargs and kargs["no_smooth"]:
        return edge

    if "mode" in kargs and kargs["mode"] == "close":
        edge = ndimage.binary_closing(
            edge, structure=np.ones(structure_size))
    else:
        edge = ndimage.binary_opening(
            edge, structure=np.ones(structure_size))
    return edge*1


def outedge(alpha, structure_size=(1, 1), **kargs):
    """Generate outedge map of alpha.

        Outedge is a binary image of numpy array. See innedge() for details.

        Args:
            alpha: Gray scale image in numpy array, should be normalized.
            structure_size: Size of rectangle used in open/close operation, tuple.
            **kargs: 
                no_smooth: Bool. Set true if using open/close operation to smooth edge.
                mode: "open" or "close", string. Choose to use open or close operation to smooth edge. We use open operation by default.

        Returns:
            Outedge, a binary image in numpy array.

        Raises:
            ValueError: An error occurred when alpha is not in gray scale.
    """
    return innedge(alpha, structure_size=structure_size, outedge=True, **kargs)


def bin_upscale(edge, scale=2, structure_size=(1, 1), **kargs):
    """Upscale the binary image.

        Binary image has only 1 or 0, so we use Computer Graphics to upscale it instead of interplotion. We set the uncertain pixels to 0 and smooth the whole image by close operation.

        Args:
            edge: Binary image in numpy array.
            scale: Scale to be multiplication, int.
            structure_size: Tuple of size of the structure when process close operation. We use rectangle as the structure in close operation.
            **kargs:
                no_smooth: Set true if you want the non-smoothed image.

        Returns:
            Edge image which is upscaled. Binary image.

        Raises:
            ValueError: An error occured when edge is not a binary image.
            ValueError: An error occured when scale is not a Int.
    """
    _ = _is_gray(edge)  # if not gray, raise ValueError. See _is_gray().
    if not isinstance(scale, int):
        raise ValueError("scale should be a integer.")

    def _bin_upscale(edge, scale):
        # upscale image.
        h, w = edge.shape
        new_edge = np.zeros((h*scale, w*scale))
        for i in range(h):
            for j in range(w):
                new_edge[i*scale, j*scale] = edge[i, j]
        return new_edge
    _new_edge = _bin_upscale(edge, scale)
    if "no_smooth" in kargs and kargs["no_smooth"]:
        return _new_edge

    # smooth by close operation.
    new_edge = ndimage.binary_closing(
        _new_edge, structure=np.ones(structure_size))
    return new_edge*1


def merge_edge(innedge, outedge, alpha, *args):
    """Merge innedge and outedge together with alpha.

        Because the white pixels in innedge and black pixels in outedge are assumed to be absolutely correct, so we fix the alpha through innedge and outedge images. 

        Args:
            innedge, outedge: Binary images in numpy array.
            alpha: Gray image in numpy array between 0 and 1.
            *args:
                return_field: String. Add it to args if you only want the field between innedge and outedge imags.

        Returns:
            Merged alpha in numpy array. It's in gray scale.

        Raises:
            ValueError: An error occured when one of the alpha, innedge and outedge is not in gray scale.
    """
    _ = map(_is_gray, [innedge, outedge, alpha]
            )  # if not gray, raise ValueError. See _is_gray().
    if np.max(alpha) > 1:
        alpha /= 255.  # normalize
    innedge = innedge.astype(np.bool_)
    outedge = outedge.astype(np.bool_)
    image = np.ones(alpha.shape) if "return_field" in args else alpha
    image = np.where(outedge, np.where(innedge, 1, image), 0)  # set 0 and 1
    return image


def loss_edge(alpha, **edges):
    """Calculate the loss of innedge or outedge image.

        Under expecting that the white pixels in innedge and black pixels in outedge are absolutely correct, we can calculate the loss if it's not perfect by counting the number of unqualified pixels.

        Args:
            alpha: Gray image in numpy array between 0 and 1.
            **edges: "name" = Binary images. "innedge" or "outedge" should be included in the name of keys!!!!

        Returns:
            List of the values of losss between alpha and all edge. If only one loss exists, return the value in Int.

        Raises:
            ValueError: An error occured when no edges is given.
            ValueError: An error occured when alpha or edge is not in gray scale.
            Warning: An warning occured when one of the kargs' name doesn't include "innedge" or "outedge".
    """
    _imgs = list(edges.values())
    _imgs.append(alpha)
    _ = map(_is_gray, _imgs)  # if not gray, raise ValueError. See _is_gray().
    if np.max(alpha) > 1:
        alpha /= 255.  # normalize

    def _loss_edge(name_edge):
        # name_edge is the key.
        edge_ = edges[name_edge]
        if "innedge" in name_edge:
            # innedge loss
            return np.sum((edge_*alpha != edge_))
        elif "outedge" in name_edge:
            # outedge loss
            return np.sum((edge_ | (alpha != 0) != (alpha != 0)))
        else:
            raise Warning(
                "Attention! If %s is an edge_ image, the name of this numpy array should includes 'innedge_' or 'outedge_', or it cannot be detected!!" % (name_edge))
    keys_edges = list(edges.keys())
    if len(keys_edges):
        loss = _loss_edge(keys_edges[0]) if len(
            keys_edges) == 1 else list(map(_loss_edge, keys_edges))
        return loss
    else:
        raise ValueError(
            "No damn edge is given, how the hell can I give u loss?")


def alpha_uscale(alpha, scale, interp="bicubic", **kargs):
    """Upscale alpha image in mulns' way.

        TODO(mulns):complete this comment.

        Args:
            alpha: Gray-scale image in numpy array between 0 and 1.
            scale: Upscale factor. Int.
            *args:
                tuples.

        Returns:
            upscaled alpha in numpy array, which is normalized.

        Raises:
            ValueError: An error occured when alpha is not in gray-scale.
    """
    innex = kargs["innex"] if "innex" in kargs else (3, 3)
    outex = kargs["outex"] if "outex" in kargs else (1, 1)
    innup = kargs["innup"] if "innup" in kargs else (4, 4)
    outup = kargs["outup"] if "outup" in kargs else (14, 14)
    up_shape = tuple(i*scale for i in alpha.shape)
    # Edge Extraction.
    al_innedge = innedge(alpha, structure_size=innex)
    al_outedge = outedge(alpha, structure_size=outex)
    # Edge Upscaling.
    up_innedge = bin_upscale(al_innedge, scale,
                             structure_size=innup)
    up_outedge = bin_upscale(al_outedge, scale,
                             structure_size=outup)
    # Merge with upscaled alpha
    up_alpha = misc.imresize(alpha, up_shape, interp=interp)/255.
    final_alpha = merge_edge(up_innedge, up_outedge, up_alpha)
    return final_alpha


############################################
# Alpha Matting
############################################


"""Alpha Matting Operations.

    In Alpha-Matting, image can be divided into fore-ground image, back-ground image and alpha image:

            mi = alpha * fg + (1 - alpha) * bg

        mi: merged image, fg: fore-ground image, bg: back-ground image.
    Alpha image is a singel-channel image no matter how many channels do mi and fg have. It will work on each channel. #XXX Why can't we have 3 alpha?
    So if given three of these four images, we can estimate the another one.
    We usually set the size of fg as the final size, cause fg is more important than bg image. So if bg is not right, we will resize it to fg's size by default. But fg, alpha and mi should share the same size.
"""


def gen_alpha(mi, fg, bg):
    """Generate alpha using mi, fg and bg image.

        This is a reverse transform of merge_mi():

            alpha = (mi - bg) / (fg - bg)

        mi should be in the same shape with fg. If bg's shape is incorrect, we resize it to shape of fg by default. alpha is supposed to be in one channel, but in this way alpha has three channels which are not completely same. Usually we generate alpha by CNN (or other orgrithm). This is just a reverse transform of merge_mi(). Cause we have to return a one channel alpha, we will choose first channel to return.  #XXX can we use YCbCr channel?

        Args:
            mi: merged image in numpy array.
            fg: fore-ground image, numpy array in the same shape with mi.
            bg: back-ground image in numpy array.

        Returns:
            alpha: gray-scale image in one channel, numpy array between 0 and 1.

        Raises:
            ValueError: An error occured when fg and mi are in different shape.
            Warning: An warning occured when alpha's three channel is not completely same.
    """
    if mi.shape == fg.shape:
        bg = misc.imresize(bg, fg.shape)
        alpha = (mi - bg) / (fg - bg)
        # check if all channel are same
        if sum(list(map(lambda x: np.sum(alpha-x), [alpha[:, :, i] for i in np.arange[alpha.shape[-1]]]))) == 0:
            raise Warning(
                "Alpha's channels are not completely same! Maybe fg doesn't match mi ?")
        return alpha[0]
    else:
        raise ValueError("mi image and fg should be in same shape! While input has shape of %s and %s." % (
            str(mi.shape), str(fg.shape)))


def merge_mi(fg, bg, alpha):
    """Merge fg and bg to mi by alpha.

        If given fg, bg and alpha, merged image can be generated through:

            mi = alpha * fg + (1 - alpha) * bg

        So fg and alpha should be in same height and width. Notice that their channel can be different, cause alpha is a singel channel image and works on all channels in one time. If bg image is not in the same shape with fg, we will resize it to fg's shape by default.(Together with channel).

        Args:
            alpha: a gray-scale image in numpy array between 0 and 1.
            fg: fore-ground image, numpy array in the same size (height and width) with alpha.
            bg: back-ground image in numpy array.

        Returns:
            Merged image in numpy array.

        Raises:
            ValueError: An error occured when fg and alpha are in different size (height and width).
            ValueError: An error occurred when alpha is not in gray scale.
    """
    _ = _is_gray(alpha)  # if not gray, raise ValueError. See _is_gray().
    if np.max(alpha) > 1:
        alpha /= 255.  # normalize
    if len(fg.shape) == 3:
        alpha = alpha[:, :, np.newaxis]  # expand channel dimension.
    if fg.shape[:2] == alpha.shape[:2]:
        bg = misc.imresize(bg, fg.shape)
        mi = fg * alpha + bg * (1 - alpha)
        return mi
    else:
        raise ValueError("fg and alpha should in the same height and width. While input has size of %s and %s." % (
            str(fg.shape[:2]), str(alpha.shape[:2])))


def main():
    pass


if __name__ == '__main__':
    main()
