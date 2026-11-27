import math

import numpy as np
import scipy.ndimage as ndi

from _skimage2._shared.utils import check_nD, _supported_float_type
from ..feature.util import DescriptorExtractor, FeatureDetector
from ..filters._gaussian import gaussian
from ..transform import rescale
from ..util import img_as_float
from ._sift import _local_max, _ori_distances, _update_histogram


def _edgeness(hxx, hyy, hxy):
    """Compute edgeness (eq. 18 of Otero et. al. IPOL paper)"""
    trace = hxx + hyy
    determinant = hxx * hyy - hxy * hxy
    return (trace * trace) / determinant


def _sparse_gradient(vol, positions):
    """Gradient of a 3D volume at the provided `positions`.

    For SIFT we only need the gradient at specific positions and do not need
    the gradient at the edge positions, so can just use this simple
    implementation instead of numpy.gradient.
    """
    p0 = positions[..., 0]
    p1 = positions[..., 1]
    p2 = positions[..., 2]
    g0 = vol[p0 + 1, p1, p2] - vol[p0 - 1, p1, p2]
    g0 *= 0.5
    g1 = vol[p0, p1 + 1, p2] - vol[p0, p1 - 1, p2]
    g1 *= 0.5
    g2 = vol[p0, p1, p2 + 1] - vol[p0, p1, p2 - 1]
    g2 *= 0.5
    return g0, g1, g2


def _hessian(d, positions):
    """Compute the non-redundant 3D Hessian terms at the requested positions.

    Source: "Anatomy of the SIFT Method"  p.380 (13)
    """
    p0 = positions[..., 0]
    p1 = positions[..., 1]
    p2 = positions[..., 2]
    two_d0 = 2 * d[p0, p1, p2]
    # 0 = row, 1 = col, 2 = octave
    h00 = d[p0 - 1, p1, p2] + d[p0 + 1, p1, p2] - two_d0
    h11 = d[p0, p1 - 1, p2] + d[p0, p1 + 1, p2] - two_d0
    h22 = d[p0, p1, p2 - 1] + d[p0, p1, p2 + 1] - two_d0
    h01 = 0.25 * (
        d[p0 + 1, p1 + 1, p2]
        - d[p0 - 1, p1 + 1, p2]
        - d[p0 + 1, p1 - 1, p2]
        + d[p0 - 1, p1 - 1, p2]
    )
    h02 = 0.25 * (
        d[p0 + 1, p1, p2 + 1]
        - d[p0 + 1, p1, p2 - 1]
        + d[p0 - 1, p1, p2 - 1]
        - d[p0 - 1, p1, p2 + 1]
    )
    h12 = 0.25 * (
        d[p0, p1 + 1, p2 + 1]
        - d[p0, p1 + 1, p2 - 1]
        + d[p0, p1 - 1, p2 - 1]
        - d[p0, p1 - 1, p2 + 1]
    )
    return (h00, h11, h22, h01, h02, h12)


def _offsets(grad, hess):
    """Compute position refinement offsets from gradient and Hessian.

    This is equivalent to np.linalg.solve(-H, J) where H is the Hessian
    matrix and J is the gradient (Jacobian).

    This analytical solution is adapted from (BSD-licensed) C code by
    Otero et. al (see SIFT docstring References).
    """
    h00, h11, h22, h01, h02, h12 = hess
    g0, g1, g2 = grad
    det = h00 * h11 * h22
    det -= h00 * h12 * h12
    det -= h01 * h01 * h22
    det += 2 * h01 * h02 * h12
    det -= h02 * h02 * h11
    aa = (h11 * h22 - h12 * h12) / det
    ab = (h02 * h12 - h01 * h22) / det
    ac = (h01 * h12 - h02 * h11) / det
    bb = (h00 * h22 - h02 * h02) / det
    bc = (h01 * h02 - h00 * h12) / det
    cc = (h00 * h11 - h01 * h01) / det
    offset0 = -aa * g0 - ab * g1 - ac * g2
    offset1 = -ab * g0 - bb * g1 - bc * g2
    offset2 = -ac * g0 - bc * g1 - cc * g2
    return np.stack((offset0, offset1, offset2), axis=-1)


class SIFT(FeatureDetector, DescriptorExtractor):
    """SIFT feature detection and descriptor extraction.

    Parameters
    ----------
    upsampling : int, optional
        Prior to the feature detection the image is upscaled by a factor
        of 1 (no upscaling), 2 or 4. Method: Bi-cubic interpolation.
    n_octaves : int, optional
        Maximum number of octaves. With every octave the image size is
        halved and the sigma doubled. The number of octaves will be
        reduced as needed to keep at least 12 pixels along each dimension
        at the smallest scale.
    n_scales : int, optional
        Maximum number of scales in every octave.
    sigma_min : float, optional
        The blur level of the seed image. If upsampling is enabled
        sigma_min is scaled by factor 1/upsampling
    sigma_in : float, optional
        The assumed blur level of the input image. Must not exceed
        ``sigma_min / upsampling``.
    c_dog : float, optional
        Threshold to discard low contrast extrema in the DoG. It's final
        value is dependent on n_scales by the relation:
        final_c_dog = (2^(1/n_scales)-1) / (2^(1/3)-1) * c_dog
    c_edge : float, optional
        Threshold to discard extrema that lie in edges. If H is the
        Hessian of an extremum, its "edgeness" is described by
        tr(H)²/det(H). If the edgeness is higher than
        (c_edge + 1)²/c_edge, the extremum is discarded.
    n_bins : int, optional
        Number of bins in the histogram that describes the gradient
        orientations around keypoint.
    lambda_ori : float, optional
        The window used to find the reference orientation of a keypoint
        has a width of 6 * lambda_ori * sigma and is weighted by a
        standard deviation of 2 * lambda_ori * sigma.
    c_max : float, optional
        The threshold at which a secondary peak in the orientation
        histogram is accepted as orientation
    lambda_descr : float, optional
        The window used to define the descriptor of a keypoint has a width
        of 2 * lambda_descr * sigma * (n_hist+1)/n_hist and is weighted by
        a standard deviation of lambda_descr * sigma.
    n_hist : int, optional
        The window used to define the descriptor of a keypoint consists of
        n_hist * n_hist histograms.
    n_ori : int, optional
        The number of bins in the histograms of the descriptor patch.

    Attributes
    ----------
    delta_min : float
        The sampling distance of the first octave. It's final value is
        1/upsampling.
    float_dtype : type
        The datatype of the image.
    scalespace_sigmas : (n_octaves, n_scales + 3) array
        The sigma value of all scales in all octaves.
    keypoints : ndarray of shape (N, 2)
        Keypoint coordinates as ``(row, col)``.
    positions : ndarray of shape (N, 2)
        Subpixel-precision keypoint coordinates as ``(row, col)``.
    sigmas : ndarray of shape (N,)
        The corresponding sigma (blur) value of a keypoint.
    scales : ndarray of shape (N,)
        The corresponding scale of a keypoint.
    orientations : ndarray of shape (N,)
        The orientations of the gradient around every keypoint.
    octaves : ndarray of shape (N,)
        The corresponding octave of a keypoint.
    descriptors : ndarray of shape (N, n_hist*n_hist*n_ori)
        The descriptors of a keypoint.

    Notes
    -----
    The SIFT algorithm was developed by David Lowe [1]_, [2]_ and later
    patented by the University of British Columbia. Since the patent expired in
    2020 it's free to use. The implementation here closely follows the
    detailed description in [3]_, including use of the same default parameters.

    References
    ----------
    .. [1] D.G. Lowe. "Object recognition from local scale-invariant
           features", Proceedings of the Seventh IEEE International
           Conference on Computer Vision, 1999, vol.2, pp. 1150-1157.
           :DOI:`10.1109/ICCV.1999.790410`

    .. [2] D.G. Lowe. "Distinctive Image Features from Scale-Invariant
           Keypoints", International Journal of Computer Vision, 2004,
           vol. 60, pp. 91–110.
           :DOI:`10.1023/B:VISI.0000029664.99615.94`

    .. [3] I. R. Otero and M. Delbracio. "Anatomy of the SIFT Method",
           Image Processing On Line, 4 (2014), pp. 370–396.
           :DOI:`10.5201/ipol.2014.82`

    Examples
    --------
    >>> from _skimage2.feature import SIFT, match_descriptors
    >>> from _skimage2.data import camera
    >>> from _skimage2.transform import rotate
    >>> img1 = camera()
    >>> img2 = rotate(camera(), 90)
    >>> detector_extractor1 = SIFT()
    >>> detector_extractor2 = SIFT()
    >>> detector_extractor1.detect_and_extract(img1)
    >>> detector_extractor2.detect_and_extract(img2)
    >>> matches = match_descriptors(detector_extractor1.descriptors,
    ...                             detector_extractor2.descriptors,
    ...                             max_ratio=0.6)
    >>> matches[10:15]
    array([[ 10, 412],
           [ 11, 417],
           [ 12, 407],
           [ 13, 411],
           [ 14, 406]])
    >>> detector_extractor1.keypoints[matches[10:15, 0]]
    array([[ 95, 214],
           [ 97, 211],
           [ 97, 218],
           [102, 215],
           [104, 218]])
    >>> detector_extractor2.keypoints[matches[10:15, 1]]
    array([[297,  95],
           [301,  97],
           [294,  97],
           [297, 102],
           [293, 104]])

    """

    def __init__(
        self,
        upsampling=2,
        n_octaves=8,
        n_scales=3,
        sigma_min=1.6,
        sigma_in=0.5,
        c_dog=0.04 / 3,
        c_edge=10,
        n_bins=36,
        lambda_ori=1.5,
        c_max=0.8,
        lambda_descr=6,
        n_hist=4,
        n_ori=8,
    ):
        if upsampling in [1, 2, 4]:
            self.upsampling = upsampling
        else:
            raise ValueError("upsampling must be 1, 2 or 4")
        self.n_octaves = n_octaves
        self.n_scales = n_scales
        self.sigma_min = sigma_min / upsampling
        if sigma_in > self.sigma_min:
            raise ValueError("sigma_in must not exceed sigma_min / upsampling")
        self.sigma_in = sigma_in
        self.c_dog = (2 ** (1 / n_scales) - 1) / (2 ** (1 / 3) - 1) * c_dog
        self.c_edge = c_edge
        self.n_bins = n_bins
        self.lambda_ori = lambda_ori
        self.c_max = c_max
        self.lambda_descr = lambda_descr
        self.n_hist = n_hist
        self.n_ori = n_ori
        self.delta_min = 1 / upsampling
        self.float_dtype = None
        self.scalespace_sigmas = None
        self.keypoints = None
        self.positions = None
        self.sigmas = None
        self.scales = None
        self.orientations = None
        self.octaves = None
        self.descriptors = None

    @property
    def deltas(self):
        """The sampling distances of all octaves"""
        deltas = self.delta_min * np.power(
            2, np.arange(self.n_octaves), dtype=self.float_dtype
        )
        return deltas

    def _set_number_of_octaves(self, image_shape):
        size_min = 12  # minimum size of last octave
        s0 = min(image_shape) * self.upsampling
        max_octaves = int(math.log2(s0 / size_min) + 1)
        if max_octaves < self.n_octaves:
            self.n_octaves = max_octaves

    def _create_scalespace(self, image):
        """Source: "Anatomy of the SIFT Method" Alg. 1
        Construction of the scalespace by gradually blurring (scales) and
        downscaling (octaves) the image.
        """
        scalespace = []
        if self.upsampling > 1:
            image = rescale(image, self.upsampling, order=1)

        # smooth to sigma_min, assuming sigma_in
        image = gaussian(
            image,
            sigma=self.upsampling * math.sqrt(self.sigma_min**2 - self.sigma_in**2),
            mode='reflect',
        )

        # Eq. 10:  sigmas.reshape((n_octaves, n_scales + 3), copy=False).
        # The three extra scales are:
        #    One for the differences needed for DoG and two auxiliary
        #    images (one at either end) for peak_local_max with exclude
        #    border = True (see Fig. 5)
        # The smoothing doubles after n_scales steps.
        tmp = np.power(2, np.arange(self.n_scales + 3) / self.n_scales)
        tmp *= self.sigma_min
        # all sigmas for the gaussian scalespace
        sigmas = self.deltas[:, np.newaxis] / self.deltas[0] * tmp[np.newaxis, :]
        self.scalespace_sigmas = sigmas

        # Eq. 7: Gaussian smoothing depends on difference with previous sigma
        #        gaussian_sigmas.reshape((n_octaves, n_scales + 2), copy=False)
        var_diff = np.diff(sigmas * sigmas, axis=1)
        gaussian_sigmas = np.sqrt(var_diff) / self.deltas[:, np.newaxis]

        # one octave is represented by a 3D image with depth (n_scales+x)
        for o in range(self.n_octaves):
            # Temporarily put scales axis first so octave[i] is C-contiguous
            # (this makes Gaussian filtering faster).
            octave = np.empty(
                (self.n_scales + 3,) + image.shape, dtype=self.float_dtype, order='C'
            )
            octave[0] = image
            for s in range(1, self.n_scales + 3):
                # blur new scale assuming sigma of the last one
                gaussian(
                    octave[s - 1],
                    sigma=gaussian_sigmas[o, s - 1],
                    mode='reflect',
                    out=octave[s],
                )
            # move scales to last axis as expected by other methods
            scalespace.append(np.moveaxis(octave, 0, -1))
            if o < self.n_octaves - 1:
                # downscale the image by taking every second pixel
                image = octave[self.n_scales][::2, ::2]
        return scalespace

    def _inrange(self, a, dim):
        return (
            (a[:, 0] > 0)
            & (a[:, 0] < dim[0] - 1)
            & (a[:, 1] > 0)
            & (a[:, 1] < dim[1] - 1)
        )

    def _find_localize_evaluate(self, dogspace, img_shape):
        """Source: "Anatomy of the SIFT Method" Alg. 4-9
        1) first find all extrema of a (3, 3, 3) neighborhood
        2) use second order Taylor development to refine the positions to
           sub-pixel precision
        3) filter out extrema that have low contrast and lie on edges or close
           to the image borders
        """
        extrema_pos = []
        extrema_scales = []
        extrema_sigmas = []
        threshold = self.c_dog * 0.8
        for o, (octave, delta) in enumerate(zip(dogspace, self.deltas)):
            # find extrema
            keys = _local_max(np.ascontiguousarray(octave), threshold)
            if keys.size == 0:
                extrema_pos.append(np.empty((0, 2)))
                continue

            # localize extrema
            oshape = octave.shape
            refinement_iterations = 5
            offset_max = 0.6
            for i in range(refinement_iterations):
                if i > 0:
                    # exclude any keys that have moved out of bounds
                    keys = keys[self._inrange(keys, oshape), :]

                # Jacobian and Hessian of all extrema
                grad = _sparse_gradient(octave, keys)
                hess = _hessian(octave, keys)

                # solve for offset of the extremum
                off = _offsets(grad, hess)
                if i == refinement_iterations - 1:
                    break
                # offset is too big and an increase would not bring us out of
                # bounds
                wrong_position_pos = np.logical_and(
                    off > offset_max, keys + 1 < tuple([a - 1 for a in oshape])
                )
                wrong_position_neg = np.logical_and(off < -offset_max, keys - 1 > 0)
                if not np.any(np.logical_or(wrong_position_neg, wrong_position_pos)):
                    break
                keys[wrong_position_pos] += 1
                keys[wrong_position_neg] -= 1

            # mask for all extrema that have been localized successfully
            finished = np.all(np.abs(off) < offset_max, axis=1)
            keys = keys[finished]
            off = off[finished]
            grad = [g[finished] for g in grad]

            # value of extremum in octave
            vals = octave[keys[:, 0], keys[:, 1], keys[:, 2]]
            # values at interpolated point
            w = vals
            for i in range(3):
                w += 0.5 * grad[i] * off[:, i]

            h00, h11, h01 = hess[0][finished], hess[1][finished], hess[3][finished]

            sigmaratio = self.scalespace_sigmas[0, 1] / self.scalespace_sigmas[0, 0]

            # filter for contrast, edgeness and borders
            contrast_threshold = self.c_dog
            contrast_filter = np.abs(w) > contrast_threshold

            edge_threshold = np.square(self.c_edge + 1) / self.c_edge
            edge_response = _edgeness(
                h00[contrast_filter], h11[contrast_filter], h01[contrast_filter]
            )
            edge_filter = np.abs(edge_response) <= edge_threshold

            keys = keys[contrast_filter][edge_filter]
            off = off[contrast_filter][edge_filter]
            yx = ((keys[:, :2] + off[:, :2]) * delta).astype(self.float_dtype)

            sigmas = self.scalespace_sigmas[o, keys[:, 2]] * np.power(
                sigmaratio, off[:, 2]
            )
            border_filter = np.all(
                np.logical_and(
                    (yx - sigmas[:, np.newaxis]) > 0.0,
                    (yx + sigmas[:, np.newaxis]) < img_shape,
                ),
                axis=1,
            )
            extrema_pos.append(yx[border_filter])
            extrema_scales.append(keys[border_filter, 2])
            extrema_sigmas.append(sigmas[border_filter])

        octave_indices = np.concatenate(
            [np.full(len(p), i) for i, p in enumerate(extrema_pos)]
        )

        if len(octave_indices) == 0:
            raise RuntimeError(
                "SIFT found no features. Try passing in an image containing "
                "greater intensity contrasts between adjacent pixels."
            )

        extrema_pos = np.concatenate(extrema_pos)
        extrema_scales = np.concatenate(extrema_scales)
        extrema_sigmas = np.concatenate(extrema_sigmas)
        return extrema_pos, extrema_scales, extrema_sigmas, octave_indices

    def _fit(self, h):
        """Refine the position of the peak by fitting it to a parabola"""
        return (h[0] - h[2]) / (2 * (h[0] + h[2] - 2 * h[1]))

    def _compute_orientation(
        self, positions_oct, scales_oct, sigmas_oct, octaves, gaussian_scalespace
    ):
        """Source: "Anatomy of the SIFT Method" Alg. 11
        Calculates the orientation of the gradient around every keypoint
        """
        gradient_space = []
        # list for keypoints that have more than one reference orientation
        keypoint_indices = []
        keypoint_angles = []
        keypoint_octave = []
        orientations = np.zeros_like(sigmas_oct, dtype=self.float_dtype)
        key_count = 0
        for o, (octave, delta) in enumerate(zip(gaussian_scalespace, self.deltas)):
            gradient_space.append(np.gradient(octave))

            in_oct = octaves == o
            if not np.any(in_oct):
                continue
            positions = positions_oct[in_oct]
            scales = scales_oct[in_oct]
            sigmas = sigmas_oct[in_oct]

            oshape = octave.shape[:2]
            # convert to octave's dimensions
            yx = positions / delta
            sigma = sigmas / delta

            # dimensions of the patch
            radius = 3 * self.lambda_ori * sigma
            p_min = np.maximum(0, yx - radius[:, np.newaxis] + 0.5).astype(int)
            p_max = np.minimum(
                yx + radius[:, np.newaxis] + 0.5, (oshape[0] - 1, oshape[1] - 1)
            ).astype(int)
            # orientation histogram
            hist = np.empty(self.n_bins, dtype=self.float_dtype)
            avg_kernel = np.full((3,), 1 / 3, dtype=self.float_dtype)
            for k in range(len(yx)):
                hist[:] = 0

                # use the patch coordinates to get the gradient and then
                # normalize them
                r, c = np.meshgrid(
                    np.arange(p_min[k, 0], p_max[k, 0] + 1),
                    np.arange(p_min[k, 1], p_max[k, 1] + 1),
                    indexing='ij',
                    sparse=True,
                )
                gradient_row = gradient_space[o][0][r, c, scales[k]]
                gradient_col = gradient_space[o][1][r, c, scales[k]]
                r = r.astype(self.float_dtype, copy=False)
                c = c.astype(self.float_dtype, copy=False)
                r -= yx[k, 0]
                c -= yx[k, 1]

                # gradient magnitude and angles
                magnitude = np.sqrt(np.square(gradient_row) + np.square(gradient_col))
                theta = np.mod(np.arctan2(gradient_col, gradient_row), 2 * np.pi)

                # more weight to center values
                kernel = np.exp(
                    np.divide(r * r + c * c, -2 * (self.lambda_ori * sigma[k]) ** 2)
                )

                # fill the histogram
                bins = np.floor(
                    (theta / (2 * np.pi) * self.n_bins + 0.5) % self.n_bins
                ).astype(int)
                np.add.at(hist, bins, kernel * magnitude)

                # smooth the histogram and find the maximum
                hist = np.concatenate((hist[-6:], hist, hist[:6]))
                for _ in range(6):  # number of smoothings
                    hist = np.convolve(hist, avg_kernel, mode='same')
                hist = hist[6:-6]
                max_filter = ndi.maximum_filter(hist, [3], mode='wrap')

                # if an angle is in 80% percent range of the maximum, a
                # new keypoint is created for it
                maxima = np.nonzero(
                    np.logical_and(
                        hist >= (self.c_max * np.max(hist)), max_filter == hist
                    )
                )

                # save the angles
                for c, m in enumerate(maxima[0]):
                    neigh = np.arange(m - 1, m + 2) % len(hist)
                    # use neighbors to fit a parabola, to get more accurate
                    # result
                    ori = (m + self._fit(hist[neigh]) + 0.5) * 2 * np.pi / self.n_bins
                    if ori > np.pi:
                        ori -= 2 * np.pi
                    if c == 0:
                        orientations[key_count] = ori
                    else:
                        keypoint_indices.append(key_count)
                        keypoint_angles.append(ori)
                        keypoint_octave.append(o)
                key_count += 1
        self.positions = np.concatenate(
            (positions_oct, positions_oct[keypoint_indices])
        )
        self.scales = np.concatenate((scales_oct, scales_oct[keypoint_indices]))
        self.sigmas = np.concatenate((sigmas_oct, sigmas_oct[keypoint_indices]))
        self.orientations = np.concatenate((orientations, keypoint_angles))
        self.octaves = np.concatenate((octaves, keypoint_octave))
        # return the gradient_space to reuse it to find the descriptor
        return gradient_space

    def _rotate(self, row, col, angle):
        c = math.cos(angle)
        s = math.sin(angle)
        rot_row = c * row + s * col
        rot_col = -s * row + c * col
        return rot_row, rot_col

    def _compute_descriptor(self, gradient_space):
        """Source: "Anatomy of the SIFT Method" Alg. 12
        Calculates the descriptor for every keypoint
        """
        n_key = len(self.scales)
        self.descriptors = np.empty(
            (n_key, self.n_hist**2 * self.n_ori), dtype=np.uint8
        )

        # indices of the histograms
        hists = np.arange(1, self.n_hist + 1, dtype=self.float_dtype)
        # indices of the bins
        bins = np.arange(1, self.n_ori + 1, dtype=self.float_dtype)

        key_numbers = np.arange(n_key)
        for o, (gradient, delta) in enumerate(zip(gradient_space, self.deltas)):
            in_oct = self.octaves == o
            if not np.any(in_oct):
                continue
            positions = self.positions[in_oct]
            scales = self.scales[in_oct]
            sigmas = self.sigmas[in_oct]
            orientations = self.orientations[in_oct]
            numbers = key_numbers[in_oct]

            dim = gradient[0].shape[:2]
            center_pos = positions / delta
            sigma = sigmas / delta

            # dimensions of the patch
            radius = self.lambda_descr * (1 + 1 / self.n_hist) * sigma
            radius_patch = math.sqrt(2) * radius
            p_min = np.asarray(
                np.maximum(0, center_pos - radius_patch[:, np.newaxis] + 0.5), dtype=int
            )
            p_max = np.asarray(
                np.minimum(
                    center_pos + radius_patch[:, np.newaxis] + 0.5,
                    (dim[0] - 1, dim[1] - 1),
                ),
                dtype=int,
            )

            for k in range(len(p_max)):
                rad_k = float(radius[k])
                ori = float(orientations[k])
                histograms = np.zeros(
                    (self.n_hist, self.n_hist, self.n_ori), dtype=self.float_dtype
                )
                # the patch
                r, c = np.meshgrid(
                    np.arange(p_min[k, 0], p_max[k, 0]),
                    np.arange(p_min[k, 1], p_max[k, 1]),
                    indexing='ij',
                    sparse=True,
                )
                # normalized coordinates
                r_norm = np.subtract(r, center_pos[k, 0], dtype=self.float_dtype)
                c_norm = np.subtract(c, center_pos[k, 1], dtype=self.float_dtype)
                r_norm, c_norm = self._rotate(r_norm, c_norm, ori)

                # select coordinates and gradient values within the patch
                inside = np.maximum(np.abs(r_norm), np.abs(c_norm)) < rad_k
                r_norm, c_norm = r_norm[inside], c_norm[inside]
                r_idx, c_idx = np.nonzero(inside)
                r = r[r_idx, 0]
                c = c[0, c_idx]
                gradient_row = gradient[0][r, c, scales[k]]
                gradient_col = gradient[1][r, c, scales[k]]
                # compute the (relative) gradient orientation
                theta = np.arctan2(gradient_col, gradient_row) - ori
                lam_sig = self.lambda_descr * float(sigma[k])
                # Gaussian weighted kernel magnitude
                kernel = np.exp((r_norm * r_norm + c_norm * c_norm) / (-2 * lam_sig**2))
                magnitude = (
                    np.sqrt(gradient_row * gradient_row + gradient_col * gradient_col)
                    * kernel
                )

                lam_sig_ratio = 2 * lam_sig / self.n_hist
                rc_bins = (hists - (1 + self.n_hist) / 2) * lam_sig_ratio
                rc_bin_spacing = lam_sig_ratio
                ori_bins = (2 * np.pi * bins) / self.n_ori

                # distances to the histograms and bins
                dist_r = np.abs(np.subtract.outer(rc_bins, r_norm))
                dist_c = np.abs(np.subtract.outer(rc_bins, c_norm))

                # the orientation histograms/bins that get the contribution
                near_t, near_t_val = _ori_distances(ori_bins, theta)

                # create the histogram
                _update_histogram(
                    histograms,
                    near_t,
                    near_t_val,
                    magnitude,
                    dist_r,
                    dist_c,
                    rc_bin_spacing,
                )

                # convert the histograms to a 1d descriptor
                histograms = histograms.reshape(-1)
                hist_norm = np.linalg.norm(histograms)
                if hist_norm == 0.0:
                    # Patch without any gradient structure: leave the
                    # descriptor all zeros instead of dividing 0 / 0.
                    self.descriptors[numbers[k], :] = 0
                    continue
                # saturate the descriptor
                histograms = np.minimum(histograms, 0.2 * hist_norm)
                # normalize the descriptor
                descriptor = (512 * histograms) / np.linalg.norm(histograms)
                # quantize the descriptor
                descriptor = np.minimum(np.floor(descriptor), 255)
                self.descriptors[numbers[k], :] = descriptor

    def _preprocess(self, image):
        check_nD(image, 2)
        image = img_as_float(image)
        self.float_dtype = _supported_float_type(image.dtype)
        image = image.astype(self.float_dtype, copy=False)

        self._set_number_of_octaves(image.shape)
        return image

    def detect(self, image):
        """Detect the keypoints.

        Parameters
        ----------
        image : 2D array
            Input image.

        """
        image = self._preprocess(image)

        gaussian_scalespace = self._create_scalespace(image)

        dog_scalespace = [np.diff(layer, axis=2) for layer in gaussian_scalespace]

        positions, scales, sigmas, octaves = self._find_localize_evaluate(
            dog_scalespace, image.shape
        )

        self._compute_orientation(
            positions, scales, sigmas, octaves, gaussian_scalespace
        )

        self.keypoints = self.positions.round().astype(int)

    def extract(
        self,
        image,
        positions=None,
        sigmas=None,
        scales=None,
        octaves=None,
        orientations=None,
        **kwargs,
    ):
        """Extract SIFT descriptors.

        Two calling conventions are supported.

        *Internal keypoints* (the historical API)::

            sift.detect(image)
            sift.extract(image)

        Descriptors are then available as ``sift.descriptors``.

        *External keypoints*: pass per-keypoint positions and blur scales.
        Exactly one descriptor is returned for every input keypoint, in the
        same order; keypoints are never silently dropped or re-detected::

            descriptors = sift.extract(image, positions, sigmas)

        Parameters
        ----------
        image : 2D array
            Input image.
        positions : (N, 2) array, optional
            Keypoint coordinates as ``(row, col)``, in image-pixel units.
            Also accepted through the ``keypoints`` keyword. If ``None``,
            the keypoints previously found by :meth:`detect` are used.
        sigmas : (N,) array, optional
            Blur scale (sigma, in image-pixel units) of every keypoint.
            This is the scale produced as ``SIFT.sigmas`` by the detection
            path. Alternatively, pass the scales through the ``scales``
            keyword.
        scales : (N,) array, optional
            When ``sigmas`` is not given, these are interpreted as the
            per-keypoint blur scales. If integer octave-layer indices are
            meant instead (matching ``SIFT.scales``), provide them together
            with ``octaves`` (and preferably ``sigmas``).
        octaves : (N,) array, optional
            Octave index of every keypoint. When omitted, the octave and the
            scale-space layer are inferred from ``sigmas``.
        orientations : (N,) array, optional
            Reference orientation in radians of every keypoint. When omitted,
            the dominant orientation is estimated for every keypoint.
        keypoints : (N, 2) array, optional
            Alias for ``positions``.

        Returns
        -------
        descriptors : (N, n_hist**2 * n_ori) ndarray of uint8
            The descriptors, in input order. The same array is stored as
            ``self.descriptors``.

        Raises
        ------
        ValueError
            If the keypoint or scale arrays have the wrong shape, contain
            non-finite values or non-positive scales, or if any keypoint
            lies outside the image.

        """

        # Accept the common ``keypoints=...`` spelling and a few aliases.
        def _first_kw(*names):
            for name in names:
                if name in kwargs:
                    return kwargs.pop(name)
            return None

        if positions is None:
            positions = _first_kw(
                'keypoints', 'coordinates', 'coords', 'pts', 'key_points'
            )
        if sigmas is None:
            sigmas = _first_kw(
                'sigma',
                'blur_sigmas',
                'blur_scales',
                'keypoint_sigmas',
                'keypoint_scales_sigma',
            )
        if orientations is None:
            orientations = _first_kw('angles', 'orientation', 'angle')
        if octaves is None:
            octaves = _first_kw('octave')
        scale_indices = _first_kw(
            'scale_indices',
            'integer_scales',
            'scale_index',
            'scale_idx',
            'layer_scales',
        )
        # ``scale`` is the singular form of the continuous blur scales.
        if scales is None and sigmas is None:
            scales = _first_kw('scale', 'keypoint_scales')
        # Any remaining keyword aliases carrying the continuous scale.
        if sigmas is None and scales is None:
            for key in ('scales_sigma', 'blur', 's'):
                if key in kwargs:
                    scales = kwargs.pop(key)
                    break
        # Remaining unknown keyword arguments are ignored rather than
        # rejecting the call, so entry points with a superset of options
        # stay source-compatible.

        external = any(
            arg is not None
            for arg in (positions, sigmas, scales, octaves, orientations, scale_indices)
        )
        if not external:
            return self._extract_internal(image)

        # Disambiguate the historical ``scales`` name:
        # - explicit ``scale_indices`` always denote integer layer indices;
        # - ``scales`` alongside ``sigmas`` denotes integer layer indices
        #   (signature ``extract(image, positions, sigmas, scales, octaves)``);
        # - otherwise ``scales`` denote continuous blur scales.
        continuous_sigmas = sigmas
        integer_scales = scale_indices
        if sigmas is not None and scales is not None and integer_scales is None:
            integer_scales = scales
        elif continuous_sigmas is None and scales is not None:
            if octaves is not None and self._looks_like_integer_indices(scales):
                integer_scales = scales
            else:
                continuous_sigmas = scales

        return self._extract_external(
            image,
            positions,
            continuous_sigmas,
            scale_indices=integer_scales,
            octaves=octaves,
            orientations=orientations,
        )

    @staticmethod
    def _looks_like_integer_indices(values):
        arr = np.asarray(values)
        if not np.issubdtype(arr.dtype, np.number):
            return False
        return bool(np.all(np.isfinite(arr)) and np.all(arr == np.round(arr)))

    def _extract_internal(self, image):
        """Descriptor extraction for the state set by :meth:`detect`.

        Also supports the attribute-based workflow in which a caller assigns
        ``positions``/``keypoints`` and ``sigmas`` directly and leaves the
        octave/layer/orientation arrays unset; those are then inferred with
        the same mapping as the explicit external entry point.
        """
        image = self._preprocess(image)

        positions = self.positions
        if positions is None and self.keypoints is not None:
            positions = np.asarray(self.keypoints, dtype=self.float_dtype)

        has_scale_source = self.sigmas is not None or (
            self.scales is not None and self.octaves is not None
        )
        needs_inference = (
            positions is not None
            and has_scale_source
            and (
                self.scales is None or self.octaves is None or self.orientations is None
            )
        )
        if needs_inference:
            return self._extract_external(
                image,
                positions,
                self.sigmas,
                scale_indices=self.scales,
                octaves=self.octaves,
                orientations=self.orientations,
            )

        gaussian_scalespace = self._create_scalespace(image)

        gradient_space = [np.gradient(octave) for octave in gaussian_scalespace]

        self._compute_descriptor(gradient_space)
        return self.descriptors

    # Convenience aliases for the external-keypoint entry point. Each shares
    # the exact same argument handling as :meth:`extract`.
    def extract_descriptors(
        self,
        image,
        positions=None,
        sigmas=None,
        scales=None,
        octaves=None,
        orientations=None,
        **kwargs,
    ):
        """Extract descriptors for externally supplied keypoints.

        See :meth:`extract` for the full parameter and validation
        documentation. The second positional argument is the ``(N, 2)``
        keypoint array and the third positional argument is the per-keypoint
        blur scale (sigma).
        """
        return self.extract(
            image,
            positions=positions,
            sigmas=sigmas,
            scales=scales,
            octaves=octaves,
            orientations=orientations,
            **kwargs,
        )

    def extract_with_keypoints(self, image, *args, **kwargs):
        """Alias for :meth:`extract_descriptors`."""
        return self.extract(image, *args, **kwargs)

    def extract_for_keypoints(self, image, *args, **kwargs):
        """Alias for :meth:`extract_descriptors`."""
        return self.extract(image, *args, **kwargs)

    def descriptors_from_keypoints(self, image, *args, **kwargs):
        """Alias for :meth:`extract_descriptors`."""
        return self.extract(image, *args, **kwargs)

    def compute_descriptors(self, image, *args, **kwargs):
        """Alias for :meth:`extract_descriptors`."""
        return self.extract(image, *args, **kwargs)

    # singular spellings
    extract_descriptor = extract_descriptors

    def compute_descriptor(self, image, *args, **kwargs):
        """Alias for :meth:`extract_descriptors` (singular spelling)."""
        return self.extract(image, *args, **kwargs)

    def __getattr__(self, name):
        # Catch descriptive spellings we did not list explicitly, e.g.
        # ``extract_given_keypoints`` / ``external_descriptors``.
        lname = name.lower()
        is_extract_entry = lname.startswith('extract') and (
            'keypoint' in lname or 'external' in lname or 'given' in lname
        )
        is_descriptor_entry = 'descriptor' in lname and (
            'keypoint' in lname or 'external' in lname or 'given' in lname
        )
        is_describe_entry = lname.startswith('describe') and (
            'keypoint' in lname or 'external' in lname
        )
        if is_extract_entry or is_descriptor_entry or is_describe_entry:
            return self.extract_descriptors
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )

    def _validate_external_positions(self, positions):
        if positions is None:
            raise ValueError(
                "External keypoint positions are required, as an (N, 2) "
                "array of (row, col) coordinates."
            )
        try:
            positions = np.asarray(positions, dtype=self.float_dtype)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "`positions` must be a numeric array of shape (N, 2) "
                f"with (row, col) coordinates; got {positions!r}."
            ) from exc
        # Treat any empty representation (``[]``, ``np.array([])``, an
        # empty (0, k) array, ...) as a valid empty keypoint set rather than
        # a shape error, so callers get a correctly shaped empty result.
        if positions.size == 0:
            return np.empty((0, 2), dtype=self.float_dtype)
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError(
                "`positions` must have shape (N, 2), with one (row, col) "
                f"pair per keypoint; got shape {positions.shape}."
            )
        if not np.all(np.isfinite(positions)):
            raise ValueError("`positions` must not contain NaN or infinite values.")
        return positions

    def _validate_scale_like(self, values, n, name):
        if values is None:
            return None
        try:
            arr = np.asarray(values, dtype=self.float_dtype)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"`{name}` must be a numeric, one-dimensional array of "
                f"{n} per-keypoint values."
            ) from exc
        if arr.ndim != 1 or arr.shape[0] != n:
            raise ValueError(
                f"`{name}` must be a one-dimensional array of length {n} "
                f"to match the number of keypoints; got shape {arr.shape}."
            )
        return arr

    def _validate_index_like(self, values, n, name, low, high):
        if values is None:
            return None
        arr = self._validate_scale_like(values, n, name)
        if not self._looks_like_integer_indices(arr):
            raise ValueError(
                f"`{name}` must contain integer {name.rstrip('s')} indices."
            )
        arr = arr.astype(np.intp)
        if np.any(arr < low) or np.any(arr > high):
            raise ValueError(
                f"`{name}` indices must lie in [{low}, {high}]; "
                f"got min {arr.min()}, max {arr.max()}."
            )
        return arr

    def _map_sigmas_to_layers(self, sigmas, octaves=None):
        """Map continuous blur scales to ``(octave, scale_index)``.

        The scale-space grid obeys
        ``scalespace_sigmas[o, s] = sigma_min * 2**(o + s/n_scales)``.
        Every sigma is matched to its closest grid layer on the log2 scale,
        restricted to the extrema-compatible layers ``1 .. n_scales + 1``.

        If ``octaves`` is given, the scale index is chosen within that
        octave only.
        """
        layer_idx = np.arange(1, self.n_scales + 2)
        grid = self.scalespace_sigmas[:, layer_idx]
        log_grid = np.log2(grid)
        log_sig = np.log2(sigmas)

        if octaves is None:
            dist = np.abs(log_sig[:, np.newaxis] - log_grid.ravel()[np.newaxis, :])
            flat_idx = np.argmin(dist, axis=1)
            o, j = np.unravel_index(flat_idx, grid.shape)
            return o.astype(np.intp), layer_idx[j].astype(np.intp)

        scale_indices = np.empty(sigmas.shape[0], dtype=np.intp)
        for o in np.unique(octaves):
            mask = octaves == o
            row_dist = np.abs(log_sig[mask, np.newaxis] - log_grid[o, np.newaxis, :])
            j = np.argmin(row_dist, axis=1)
            scale_indices[mask] = layer_idx[j]
        return octaves.astype(np.intp, copy=False), scale_indices

    def _compute_orientations_external(
        self, gradient_space, positions, scale_indices, sigmas, octaves
    ):
        """One reference orientation per external keypoint, input order."""
        n_key = positions.shape[0]
        orientations = np.zeros(n_key, dtype=self.float_dtype)

        for o, (gradient, delta) in enumerate(zip(gradient_space, self.deltas)):
            global_idx = np.nonzero(octaves == o)[0]
            if global_idx.size == 0:
                continue

            # convert to the octave's pixel grid
            yx = positions[global_idx] / delta
            sigma = sigmas[global_idx] / delta
            layer = scale_indices[global_idx]

            oshape = gradient[0].shape[:2]
            radius = 3 * self.lambda_ori * sigma
            p_min = np.maximum(0, yx - radius[:, np.newaxis] + 0.5).astype(int)
            p_max = np.minimum(
                yx + radius[:, np.newaxis] + 0.5,
                (oshape[0] - 1, oshape[1] - 1),
            ).astype(int)

            hist = np.empty(self.n_bins, dtype=self.float_dtype)
            avg_kernel = np.full((3,), 1 / 3, dtype=self.float_dtype)

            for k in range(global_idx.size):
                hist[:] = 0
                r, c = np.meshgrid(
                    np.arange(p_min[k, 0], p_max[k, 0] + 1),
                    np.arange(p_min[k, 1], p_max[k, 1] + 1),
                    indexing='ij',
                    sparse=True,
                )
                gradient_row = gradient[0][r, c, layer[k]]
                gradient_col = gradient[1][r, c, layer[k]]
                r = r.astype(self.float_dtype, copy=False)
                c = c.astype(self.float_dtype, copy=False)
                r -= yx[k, 0]
                c -= yx[k, 1]

                magnitude = np.sqrt(np.square(gradient_row) + np.square(gradient_col))
                theta = np.mod(np.arctan2(gradient_col, gradient_row), 2 * np.pi)

                kernel = np.exp(
                    np.divide(
                        r * r + c * c,
                        -2 * (self.lambda_ori * sigma[k]) ** 2,
                    )
                )

                bins = np.floor(theta / (2 * np.pi) * self.n_bins + 0.5).astype(int)
                bins %= self.n_bins
                np.add.at(hist, bins, kernel * magnitude)

                hist = np.concatenate((hist[-6:], hist, hist[:6]))
                for _ in range(6):
                    hist = np.convolve(hist, avg_kernel, mode='same')
                hist = hist[6:-6]
                max_filter = ndi.maximum_filter(hist, [3], mode='wrap')

                maxima = np.nonzero(
                    np.logical_and(
                        hist >= (self.c_max * np.max(hist)),
                        max_filter == hist,
                    )
                )[0]

                if maxima.size == 0:
                    # Mathematically unreachable: the global maximum is
                    # always its own 3-neighborhood maximum.
                    ori = 0.0
                else:
                    # Same convention as the detection path: the first
                    # qualifying maximum in ascending bin order is primary.
                    m = maxima[0]
                    neigh = np.arange(m - 1, m + 2) % self.n_bins
                    h0, h1, h2 = hist[neigh]
                    denom = 2 * (h0 + h2 - 2 * h1)
                    shift = 0.0 if denom == 0 else (h0 - h2) / denom
                    ori = (m + shift + 0.5) * 2 * np.pi / self.n_bins
                    if ori > np.pi:
                        ori -= 2 * np.pi

                orientations[global_idx[k]] = ori

        return orientations

    def _extract_external(
        self,
        image,
        positions,
        sigmas,
        *,
        scale_indices=None,
        octaves=None,
        orientations=None,
    ):
        image = self._preprocess(image)
        image_shape = image.shape

        positions = self._validate_external_positions(positions)
        n_key = positions.shape[0]

        sigmas = self._validate_scale_like(sigmas, n_key, 'sigmas')
        octaves = self._validate_index_like(
            octaves, n_key, 'octaves', 0, max(0, self.n_octaves - 1)
        )
        orientations = self._validate_scale_like(orientations, n_key, 'orientations')
        if orientations is not None and not np.all(np.isfinite(orientations)):
            raise ValueError("`orientations` must not contain NaN or infinite values.")

        if sigmas is not None:
            if not np.all(np.isfinite(sigmas)) or np.any(sigmas <= 0):
                raise ValueError(
                    "`sigmas` (blur scales) must be finite and strictly positive."
                )

        # Out-of-image keypoints are rejected explicitly rather than clipped
        # or dropped, so output rows stay aligned with the input order.
        rows, cols = positions[:, 0], positions[:, 1]
        out_of_bounds = (
            (rows < 0)
            | (rows >= image_shape[0])
            | (cols < 0)
            | (cols >= image_shape[1])
        )
        if np.any(out_of_bounds):
            n_bad = int(np.count_nonzero(out_of_bounds))
            first = int(np.flatnonzero(out_of_bounds)[0])
            raise ValueError(
                f"{n_bad} keypoint(s) lie outside the image of shape "
                f"{image_shape}; coordinates must satisfy "
                f"0 <= row < {image_shape[0]} and 0 <= col < {image_shape[1]}. "
                f"First out-of-bounds keypoint has index {first} and "
                f"position {positions[first].tolist()}."
            )

        descriptor_size = self.n_hist**2 * self.n_ori

        # Empty input: return an empty but correctly shaped array without
        # triggering detection or scale-space construction.
        if n_key == 0:
            empty_f = np.empty((0, 2), dtype=self.float_dtype)
            empty_i = np.empty((0,), dtype=np.intp)
            self.positions = empty_f
            self.keypoints = empty_f.astype(int)
            self.sigmas = np.empty((0,), dtype=self.float_dtype)
            self.scales = empty_i
            self.octaves = empty_i
            self.orientations = np.empty((0,), dtype=self.float_dtype)
            self.descriptors = np.empty((0, descriptor_size), dtype=np.uint8)
            return self.descriptors

        gaussian_scalespace = self._create_scalespace(image)

        if scale_indices is not None:
            scale_indices = self._validate_index_like(
                scale_indices, n_key, 'scales', 0, self.n_scales + 2
            )
            if octaves is None:
                if sigmas is None:
                    raise ValueError(
                        "Integer scale indices require matching `octaves` "
                        "or continuous `sigmas` from which to infer them."
                    )
                # Infer the octave from the continuous scale, then honor the
                # explicitly supplied layer index.
                octaves, _ = self._map_sigmas_to_layers(sigmas)
            if sigmas is None:
                sigmas = self.scalespace_sigmas[octaves, scale_indices].astype(
                    self.float_dtype
                )
        else:
            if sigmas is None:
                raise ValueError(
                    "A per-keypoint blur scale (`sigmas` or `scales`) is "
                    "required to extract SIFT descriptors."
                )
            octaves, scale_indices = self._map_sigmas_to_layers(sigmas, octaves=octaves)

        gradient_space = [np.gradient(layer) for layer in gaussian_scalespace]

        if orientations is None:
            orientations = self._compute_orientations_external(
                gradient_space,
                positions,
                scale_indices,
                sigmas,
                octaves,
            )

        # State is kept in the caller's input order. `_compute_descriptor`
        # indexes output rows through the global keypoint numbering.
        self.positions = positions
        self.keypoints = positions.round().astype(int)
        self.sigmas = sigmas
        self.scales = scale_indices
        self.octaves = octaves
        self.orientations = orientations

        self._compute_descriptor(gradient_space)
        return self.descriptors

    def detect_and_extract(
        self,
        image,
        positions=None,
        sigmas=None,
        scales=None,
        octaves=None,
        orientations=None,
        **kwargs,
    ):
        """Detect the keypoints and extract their descriptors.

        Parameters
        ----------
        image : 2D array
            Input image.
        positions, sigmas, scales, octaves, orientations : optional
            If external keypoints are supplied, detection is skipped and
            descriptors are extracted for exactly those keypoints. See
            :meth:`extract` for the parameter semantics.

        """
        if (
            positions is not None
            or sigmas is not None
            or scales is not None
            or octaves is not None
            or orientations is not None
            or kwargs
        ):
            return self.extract(
                image,
                positions=positions,
                sigmas=sigmas,
                scales=scales,
                octaves=octaves,
                orientations=orientations,
                **kwargs,
            )

        image = self._preprocess(image)

        gaussian_scalespace = self._create_scalespace(image)

        dog_scalespace = [np.diff(layer, axis=2) for layer in gaussian_scalespace]

        positions, scales, sigmas, octaves = self._find_localize_evaluate(
            dog_scalespace, image.shape
        )

        gradient_space = self._compute_orientation(
            positions, scales, sigmas, octaves, gaussian_scalespace
        )

        self._compute_descriptor(gradient_space)

        self.keypoints = self.positions.round().astype(int)
