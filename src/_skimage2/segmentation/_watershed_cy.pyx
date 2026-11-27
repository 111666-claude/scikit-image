"""watershed.pyx - cython implementation of guts of watershed
"""
from libc.math cimport sqrt
from _skimage2._shared.fused_numerics cimport np_anyint

cimport numpy as cnp
cimport cython
cnp.import_array()

ctypedef cnp.int8_t DTYPE_BOOL_t


include "heap_watershed.pxi"


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.overflowcheck(False)
cdef inline Py_ssize_t _periodic_neighbor(
        Py_ssize_t index,
        Py_ssize_t offset,
        cnp.intp_t[::1] step0,
        cnp.intp_t[::1] step1,
        cnp.intp_t[::1] strides,
        cnp.intp_t[::1] shape,
        DTYPE_BOOL_t[::1] periodic,
        Py_ssize_t neighbor_i,
) noexcept nogil:
    """Return the raveled index of a neighbor on a periodic 2D grid.

    ``index`` is the raveled index of the current pixel and ``offset`` the
    plain raveled offset to the desired neighbor.  ``step0`` and ``step1``
    hold the signed per-axis footprint step (-1, 0 or 1) of every neighbor,
    precomputed in Python so that no signed modulo is needed here.

    For every axis marked as periodic the current coordinate is obtained
    with a non-negative modulo (the raveled index and stride are positive)
    and whenever ``coordinate + step`` leaves ``[0, dim)`` the neighbor is
    shifted back by a full period *along that same axis*.  Each axis is
    handled from the original raveled index, so a step of ``-1`` at the
    start of a row wraps to the end of the *same* row instead of landing on
    the row above in the flattened array.
    """
    cdef:
        Py_ssize_t neighbor_index = index + offset
        Py_ssize_t coord
        Py_ssize_t target

    # no periodic axis: the plain raveled offset is always valid (padding
    # guarantees non-periodic border neighbors land on masked-out cells)
    if periodic.shape[0] == 0 or not (periodic[0] or
                                      (periodic.shape[0] > 1 and periodic[1])):
        return neighbor_index

    if periodic[0]:
        coord = (index // strides[0]) % shape[0]
        target = coord + step0[neighbor_i]
        if target < 0:
            neighbor_index += shape[0] * strides[0]
        elif target >= shape[0]:
            neighbor_index -= shape[0] * strides[0]

    if periodic.shape[0] > 1 and periodic[1]:
        coord = (index // strides[1]) % shape[1]
        target = coord + step1[neighbor_i]
        if target < 0:
            neighbor_index += shape[1] * strides[1]
        elif target >= shape[1]:
            neighbor_index -= shape[1] * strides[1]

    return neighbor_index


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.overflowcheck(False)
@cython.unraisable_tracebacks(False)
cdef inline cnp.float64_t _euclid_dist(Py_ssize_t pt0, Py_ssize_t pt1,
                                       cnp.intp_t[::1] strides) noexcept nogil:
    """Return the Euclidean distance between raveled points pt0 and pt1."""
    cdef cnp.float64_t result = 0
    cdef cnp.float64_t curr = 0
    for i in range(strides.shape[0]):
        curr = (pt0 // strides[i]) - (pt1 // strides[i])
        result += curr * curr
        pt0 = pt0 % strides[i]
        pt1 = pt1 % strides[i]
    return sqrt(result)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.unraisable_tracebacks(False)
cdef inline DTYPE_BOOL_t _diff_neighbors(np_anyint[::1] output,
                                         cnp.intp_t[::1] structure,
                                         DTYPE_BOOL_t[::1] mask,
                                         cnp.intp_t[::1] step0,
                                         cnp.intp_t[::1] step1,
                                         cnp.intp_t[::1] strides,
                                         cnp.intp_t[::1] shape,
                                         DTYPE_BOOL_t[::1] periodic,
                                         Py_ssize_t index,
                                         np_anyint label,
                                         ) noexcept nogil:
    """
    Return ``True`` and set ``mask[index]`` to ``False`` if the neighbors of
    ``index`` (as given by the offsets in ``structure``) have more than one
    distinct nonzero label.
    """
    cdef:
        Py_ssize_t i, neighbor_index
        np_anyint neighbor_label
        Py_ssize_t nneighbors = structure.shape[0]

    if not mask[index]:
        return True

    for i in range(nneighbors):
        neighbor_index = _periodic_neighbor(
            index, structure[i], step0, step1, strides, shape, periodic, i
        )
        if mask[neighbor_index]:  # neighbor not a watershed line
            neighbor_label = output[neighbor_index]
            if neighbor_label and neighbor_label != label:
                mask[index] = False
                return True
    return False

@cython.boundscheck(False)
@cython.wraparound(False)
def watershed_raveled(cnp.float64_t[::1] image,
                      cnp.intp_t[::1] marker_locations,
                      cnp.intp_t[::1] structure,
                      DTYPE_BOOL_t[::1] mask,
                      cnp.intp_t[::1] strides,
                      cnp.intp_t[::1] shape,
                      DTYPE_BOOL_t[::1] periodic,
                      cnp.intp_t[::1] step0,
                      cnp.intp_t[::1] step1,
                      cnp.float64_t compactness,
                      np_anyint[::1] output,
                      DTYPE_BOOL_t wsl):
    """Perform watershed algorithm using a raveled image and neighborhood.

    Parameters
    ----------

    image : array of float
        The flattened image pixels.
    marker_locations : array of int
        The raveled coordinates of the initial markers (aka seeds) for the
        watershed. NOTE: these should *all* point to nonzero entries in the
        output, or the algorithm will never terminate and blow up your memory!
    structure : array of int
        A list of coordinate offsets to compute the raveled coordinates of each
        neighbor from the raveled coordinates of the current pixel.
    mask : array of int
        An array of the same shape as `image` where each pixel contains a
        nonzero value if it is to be considered for flooding with watershed,
        zero otherwise. NOTE: it is *essential* that the border pixels (those
        with neighbors falling outside the volume) are all set to zero, or
        segfaults could occur. Along axes marked as periodic in `periodic`
        the volume is not padded, so wrapping is handled explicitly instead.
    strides : array of int
        An array representing the number of steps to move along each dimension.
        This is used in computing the Euclidean distance between raveled
        indices.
    shape : array of int
        The shape of the (possibly padded) raveled volume, one entry per
        dimension.  Together with `periodic` this is used to wrap neighbors
        around periodic axes.
    periodic : array of bool
        One boolean per dimension.  Entries set to True mark the
        corresponding dimension as periodic, so that neighbors that would
        fall outside the volume along that dimension are wrapped to the
        opposite side.  An empty array disables periodic handling entirely.
    step0, step1 : array of int
        Signed per-axis footprint step (-1, 0 or 1) of every entry in
        `structure`, for the first and second axis respectively.  Used to
        wrap neighbors around periodic axes without signed integer modulo.
    compactness : float
        A value greater than 0 implements the compact watershed algorithm
        (see .py file).
    output : array of int
        The output array, which must already contain nonzero entries at all the
        seed locations.
    wsl : bool
        Parameter indicating whether the watershed line is calculated.
        If wsl is set to True, the watershed line is calculated.
    """
    cdef Heapitem elem
    cdef Heapitem new_elem
    cdef Py_ssize_t nneighbors = structure.shape[0]
    cdef Py_ssize_t i = 0
    cdef Py_ssize_t age = 1
    cdef Py_ssize_t index = 0
    cdef Py_ssize_t neighbor_index = 0
    cdef DTYPE_BOOL_t compact = (compactness > 0)

    cdef Heap *hp = <Heap *> heap_from_numpy2()

    with nogil:
        for i in range(marker_locations.shape[0]):
            index = marker_locations[i]
            elem.value = image[index]
            elem.age = 0
            elem.index = index
            elem.source = index
            heappush(hp, &elem)

        while hp.items > 0:
            heappop(hp, &elem)

            if compact or wsl:
                # in the compact case, we need to label pixels as they come off
                # the heap, because the same pixel can be pushed twice, *and* the
                # later push can have lower cost because of the compactness.
                #
                # In the case of preserving watershed lines, a similar argument
                # applies: we can only observe that all neighbors have been labeled
                # as the pixel comes off the heap. Trying to do so at push time
                # is a bug.
                if output[elem.index] and elem.index != elem.source:
                    # non-marker, already visited from another neighbor
                    continue

                # when `wsl` is `True`, label is only set for pixels without a neighbor of different label
                # NOTE: `_diff_neighbors` sets `mask[elem.index]` to `False` if
                #        neighbor has different label
                if compact or not _diff_neighbors(
                        output, structure, mask, step0, step1, strides, shape,
                        periodic, elem.index, output[elem.source]):
                    output[elem.index] = output[elem.source]

            for i in range(nneighbors):
                # get the flattened address of the neighbor, wrapping around
                # periodic axes where the step would leave the volume
                neighbor_index = _periodic_neighbor(
                    elem.index, structure[i], step0, step1, strides, shape,
                    periodic, i
                )

                if not mask[neighbor_index]:
                    # this branch includes basin boundaries, aka watershed lines
                    # neighbor is not in mask
                    continue

                if output[neighbor_index]:
                    # pre-labeled neighbor is not added to the queue.
                    continue

                age += 1
                new_elem.value = image[neighbor_index]
                if compact:
                    new_elem.value += (compactness *
                                       _euclid_dist(neighbor_index, elem.source,
                                                    strides))
                elif not wsl:
                    # in the simplest watershed case (no compactness and no
                    # watershed lines), we can label a pixel at the time that
                    # we push it onto the heap, because it can't be reached with
                    # lower cost later.
                    # This results in a very significant performance gain, see:
                    # https://github.com/scikit-image/scikit-image/issues/2636
                    output[neighbor_index] = output[elem.index]
                new_elem.age = age
                new_elem.index = neighbor_index
                new_elem.source = elem.source

                # The cost adding a neighbor should be at least the cost of its
                # originating pixel, effectively its marker. This leads to "fairer"
                # outcomes in edge cases. It prevents one marker from conquering a basin
                # against other markers with a similar claim, just because it spills
                # into the basin one step earlier. Instead `age` will distribute the
                # basin more evenly between contesting markers with the same cost.
                if new_elem.value < elem.value:
                    new_elem.value = elem.value

                heappush(hp, &new_elem)

    heap_done(hp)
