import math

import numpy as np

from ..filters._gaussian import gaussian
from _skimage2._shared.utils import convert_to_float
from ._warps import resize


def _smooth(image, sigma, mode, cval, channel_axis):
    """Return image with each channel smoothed by the Gaussian filter."""
    smoothed = np.empty_like(image)

    # apply Gaussian filter to all channels independently
    if channel_axis is not None:
        # can rely on gaussian to insert a 0 entry at channel_axis
        channel_axis = channel_axis % image.ndim
        sigma = (sigma,) * (image.ndim - 1)
    else:
        channel_axis = None
    gaussian(
        image,
        sigma=sigma,
        out=smoothed,
        mode=mode,
        cval=cval,
        channel_axis=channel_axis,
    )
    return smoothed


def _check_factor(factor):
    if factor <= 1:
        raise ValueError('scale factor must be greater than 1')


def _pyramid_expand(image, out_shape, sigma, order, mode, cval, channel_axis):
    """Upsample to an explicit ``out_shape`` and smooth.

    Using an explicit output shape (instead of a fixed integer factor) makes
    the operation the exact inverse counterpart of the corresponding
    reduction, including for odd-sized axes.
    """
    resized = resize(
        image,
        out_shape,
        order=order,
        mode=mode,
        cval=cval,
        anti_aliasing=False,
    )
    return _smooth(resized, sigma, mode, cval, channel_axis)


def pyramid_reduce(
    image,
    downscale=2,
    sigma=None,
    order=1,
    mode='reflect',
    cval=0,
    preserve_range=False,
    *,
    channel_axis=None,
):
    """Smooth and then downsample image.

    Parameters
    ----------
    image : ndarray
        Input image.
    downscale : float, optional
        Downscale factor.
    sigma : float, optional
        Sigma for Gaussian filter. Default is `2 * downscale / 6.0` which
        corresponds to a filter mask twice the size of the scale factor that
        covers more than 99% of the Gaussian distribution.
    order : int, optional
        Order of splines used in interpolation of downsampling. See
        `skimage.transform.warp` for detail.
    mode : {'reflect', 'constant', 'edge', 'symmetric', 'wrap'}, optional
        The mode parameter determines how the array borders are handled, where
        cval is the value when mode is equal to 'constant'.
    cval : float, optional
        Value to fill past edges of input if mode is 'constant'.
    preserve_range : bool, optional
        Whether to keep the original range of values. Otherwise, the input
        image is converted according to the conventions of `img_as_float`.
        Also see https://scikit-image.org/docs/dev/user_guide/data_types.html
    channel_axis : int or None, optional
        If None, the image is assumed to be a grayscale (single channel) image.
        Otherwise, this parameter indicates which axis of the array corresponds
        to channels.

        .. versionadded:: 0.19
           ``channel_axis`` was added in 0.19.

    Returns
    -------
    out : array
        Smoothed and downsampled float image.

    References
    ----------
    .. [1] http://persci.mit.edu/pub_pdfs/pyramid83.pdf

    """
    _check_factor(downscale)

    image = convert_to_float(image, preserve_range)
    if channel_axis is not None:
        channel_axis = channel_axis % image.ndim
        out_shape = tuple(
            math.ceil(d / float(downscale)) if ax != channel_axis else d
            for ax, d in enumerate(image.shape)
        )
    else:
        out_shape = tuple(math.ceil(d / float(downscale)) for d in image.shape)

    if sigma is None:
        # automatically determine sigma which covers > 99% of distribution
        sigma = 2 * downscale / 6.0

    smoothed = _smooth(image, sigma, mode, cval, channel_axis)
    out = resize(
        smoothed, out_shape, order=order, mode=mode, cval=cval, anti_aliasing=False
    )

    return out


def pyramid_expand(
    image,
    upscale=2,
    sigma=None,
    order=1,
    mode='reflect',
    cval=0,
    preserve_range=False,
    *,
    channel_axis=None,
):
    """Upsample and then smooth image.

    Parameters
    ----------
    image : ndarray
        Input image.
    upscale : float, optional
        Upscale factor.
    sigma : float, optional
        Sigma for Gaussian filter. Default is `2 * upscale / 6.0` which
        corresponds to a filter mask twice the size of the scale factor that
        covers more than 99% of the Gaussian distribution.
    order : int, optional
        Order of splines used in interpolation of upsampling. See
        `skimage.transform.warp` for detail.
    mode : {'reflect', 'constant', 'edge', 'symmetric', 'wrap'}, optional
        The mode parameter determines how the array borders are handled, where
        cval is the value when mode is equal to 'constant'.
    cval : float, optional
        Value to fill past edges of input if mode is 'constant'.
    preserve_range : bool, optional
        Whether to keep the original range of values. Otherwise, the input
        image is converted according to the conventions of `img_as_float`.
        Also see https://scikit-image.org/docs/dev/user_guide/data_types.html
    channel_axis : int or None, optional
        If None, the image is assumed to be a grayscale (single channel) image.
        Otherwise, this parameter indicates which axis of the array corresponds
        to channels.

        .. versionadded:: 0.19
           ``channel_axis`` was added in 0.19.

    Returns
    -------
    out : array
        Upsampled and smoothed float image.

    References
    ----------
    .. [1] http://persci.mit.edu/pub_pdfs/pyramid83.pdf

    """
    _check_factor(upscale)
    image = convert_to_float(image, preserve_range)
    if channel_axis is not None:
        channel_axis = channel_axis % image.ndim
        out_shape = tuple(
            math.ceil(upscale * d) if ax != channel_axis else d
            for ax, d in enumerate(image.shape)
        )
    else:
        out_shape = tuple(math.ceil(upscale * d) for d in image.shape)

    if sigma is None:
        # automatically determine sigma which covers > 99% of distribution
        sigma = 2 * upscale / 6.0

    out = _pyramid_expand(image, out_shape, sigma, order, mode, cval, channel_axis)

    return out


def pyramid_gaussian(
    image,
    max_layer=-1,
    downscale=2,
    sigma=None,
    order=1,
    mode='reflect',
    cval=0,
    preserve_range=False,
    *,
    channel_axis=None,
):
    """Yield images of the Gaussian pyramid formed by the input image.

    Recursively applies the `pyramid_reduce` function to the image, and yields
    the downscaled images.

    Note that the first image of the pyramid will be the original, unscaled
    image. The total number of images is `max_layer + 1`. In case all layers
    are computed, the last image is either a one-pixel image or the image where
    the reduction does not change its shape.

    Parameters
    ----------
    image : ndarray
        Input image.
    max_layer : int, optional
        Number of layers for the pyramid. 0th layer is the original image.
        Default is -1 which builds all possible layers.
    downscale : float, optional
        Downscale factor.
    sigma : float, optional
        Sigma for Gaussian filter. Default is `2 * downscale / 6.0` which
        corresponds to a filter mask twice the size of the scale factor that
        covers more than 99% of the Gaussian distribution.
    order : int, optional
        Order of splines used in interpolation of downsampling. See
        `skimage.transform.warp` for detail.
    mode : {'reflect', 'constant', 'edge', 'symmetric', 'wrap'}, optional
        The mode parameter determines how the array borders are handled, where
        cval is the value when mode is equal to 'constant'.
    cval : float, optional
        Value to fill past edges of input if mode is 'constant'.
    preserve_range : bool, optional
        Whether to keep the original range of values. Otherwise, the input
        image is converted according to the conventions of `img_as_float`.
        Also see https://scikit-image.org/docs/dev/user_guide/data_types.html
    channel_axis : int or None, optional
        If None, the image is assumed to be a grayscale (single channel) image.
        Otherwise, this parameter indicates which axis of the array corresponds
        to channels.

        .. versionadded:: 0.19
           ``channel_axis`` was added in 0.19.

    Returns
    -------
    pyramid : generator
        Generator yielding pyramid layers as float images.

    References
    ----------
    .. [1] http://persci.mit.edu/pub_pdfs/pyramid83.pdf

    """
    _check_factor(downscale)

    # cast to float for consistent data type in pyramid
    image = convert_to_float(image, preserve_range)

    layer = 0
    current_shape = image.shape

    prev_layer_image = image
    yield image

    # build downsampled images until max_layer is reached or downscale process
    # does not change image size
    while layer != max_layer:
        layer += 1

        layer_image = pyramid_reduce(
            prev_layer_image,
            downscale,
            sigma,
            order,
            mode,
            cval,
            channel_axis=channel_axis,
        )

        prev_shape = current_shape
        prev_layer_image = layer_image
        current_shape = layer_image.shape

        # no change to previous pyramid layer
        if current_shape == prev_shape:
            break

        yield layer_image


def pyramid_laplacian(
    image,
    max_layer=-1,
    downscale=2,
    sigma=None,
    order=1,
    mode='reflect',
    cval=0,
    preserve_range=False,
    *,
    channel_axis=None,
):
    """Yield images of the laplacian pyramid formed by the input image.

    Each layer except the last contains the difference between a Gaussian
    pyramid layer and the subsequent (coarser) Gaussian layer expanded back to
    the finer layer's shape::

        layer = prev_layer - expand(next_layer)

    The last layer is the coarsest Gaussian pyramid layer itself, i.e. the
    low-frequency residual. Keeping this residual allows the original image to
    be reconstructed, up to floating-point precision, by starting from the
    last layer and recursively expanding and adding the difference layers.
    The total number of images is `max_layer + 1`. In case all layers are
    computed, the last image is either a one-pixel image or the image where
    the reduction does not change its shape.

    Parameters
    ----------
    image : ndarray
        Input image.
    max_layer : int, optional
        Number of layers for the pyramid. 0th layer is the original image.
        Default is -1 which builds all possible layers.
    downscale : float, optional
        Downscale factor.
    sigma : float, optional
        Sigma for Gaussian filter. Default is `2 * downscale / 6.0` which
        corresponds to a filter mask twice the size of the scale factor that
        covers more than 99% of the Gaussian distribution.
    order : int, optional
        Order of splines used in interpolation of downsampling. See
        `skimage.transform.warp` for detail.
    mode : {'reflect', 'constant', 'edge', 'symmetric', 'wrap'}, optional
        The mode parameter determines how the array borders are handled, where
        cval is the value when mode is equal to 'constant'.
    cval : float, optional
        Value to fill past edges of input if mode is 'constant'.
    preserve_range : bool, optional
        Whether to keep the original range of values. Otherwise, the input
        image is converted according to the conventions of `img_as_float`.
        Also see https://scikit-image.org/docs/dev/user_guide/data_types.html
    channel_axis : int or None, optional
        If None, the image is assumed to be a grayscale (single channel) image.
        Otherwise, this parameter indicates which axis of the array corresponds
        to channels.

        .. versionadded:: 0.19
           ``channel_axis`` was added in 0.19.

    Returns
    -------
    pyramid : generator
        Generator yielding pyramid layers as float images.

    References
    ----------
    .. [1] http://persci.mit.edu/pub_pdfs/pyramid83.pdf
    .. [2] http://sepwww.stanford.edu/data/media/public/sep/morgan/texturematch/paper_html/node3.html

    """
    _check_factor(downscale)

    # cast to float for consistent data type in pyramid
    image = convert_to_float(image, preserve_range)

    if sigma is None:
        # automatically determine sigma which covers > 99% of distribution
        sigma = 2 * downscale / 6.0

    current_shape = image.shape

    # Determine the spatial axes used to size the pyramid. Channel axes are
    # never downscaled.
    if channel_axis is not None:
        channel_axis = channel_axis % image.ndim
        spatial_shape = tuple(
            d for ax, d in enumerate(current_shape) if ax != channel_axis
        )
    else:
        spatial_shape = current_shape

    # Preserve the historical ``max_layer`` semantics: a negative value (-1 by
    # convention) builds all levels up to a single pixel along the largest
    # spatial axis.
    if max_layer < 0:
        max_layer = math.ceil(math.log(max(spatial_shape), downscale))

    # Gaussian pyramid levels G_0, G_1, ..., G_{max_layer}. Building the levels
    # explicitly keeps the expected shape and exactly ``max_layer + 1`` layers,
    # even when an explicitly large ``max_layer`` repeats a size-1 level.
    gaussian_levels = [image]
    layer_image = image
    for _ in range(max_layer):
        layer_image = pyramid_reduce(
            layer_image,
            downscale=downscale,
            sigma=sigma,
            order=order,
            mode=mode,
            cval=cval,
            preserve_range=True,
            channel_axis=channel_axis,
        )
        gaussian_levels.append(layer_image)

    # Each Laplacian detail layer is the difference between a fine Gaussian
    # level and the subsequent coarser level expanded back to the fine shape.
    # Expanding to the exact fine-layer shape (rather than a fixed factor) is
    # what keeps odd-sized axes invertible.
    for layer in range(max_layer):
        expanded_image = _pyramid_expand(
            gaussian_levels[layer + 1],
            gaussian_levels[layer].shape,
            sigma,
            order,
            mode,
            cval,
            channel_axis,
        )
        yield gaussian_levels[layer] - expanded_image

    # The coarsest Gaussian level is the low-frequency residual. It is not a
    # difference and provides the starting point for reconstruction.
    yield gaussian_levels[max_layer]
