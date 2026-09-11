"""
First a list of all Transforms that we consider:
Legend:
    - (X) - if already implemented but not Checked (not implemented into get_transform class)
    - (O) - if not implemented yet
    - (V) - if both implemented and in get_transform class

Phase A (fast + robust):

DCT                (X)

FFT / DST          (X)

Haar               (X)

Hadamard           (X)

Phase B (powerfull + data-driven):

bior1.5            (X) (Wavelet)

SVD                (X) (per Block-Stack)

PCA/KLT            (X) (per Block-Stack)

Phase C (specialized for Poisson, heavy cost):

Curvelet           (O) (FDCT / Curvelab wrapper)

Shearlet           (O) (ShearLab wrapper)

Stationary Wavelet (O) (à trous)

HOSVD / Tucker     (O) (Tensorly)





| Property           | DCT                     | FFT                       | DST                     | Haar             | Hadamard      | bior1.5                 | PCA / KLT         | SVD Basis         |
| ------------------ | ----------------------- | ------------------------- | ----------------------- | ---------------- | ------------- | ----------------------- | ----------------- | ----------------- |
| Support domain     | global                  | global                    | global                  | local (minimal)  | global        | local                   | data-adaptive     | data-adaptive     |
| Symmetry           | semi-symmetric          | asymmetric (complex)      | symmetric               | symmetric        | symmetric     | symmetric               | symmetric (stats) | symmetric (stats) |
| Orthogonality      | orthogonal              | orthogonal (unitary)      | orthogonal              | orthogonal       | orthogonal    | biorthogonal            | orthogonal        | orthogonal        |
| Energy compaction  | strong (smooth signals) | strong (periodic signals) | strong (smooth signals) | weak (piecewise) | moderate      | strong (natural images) | optimal (sample)  | optimal (sample)  |
| Edge handling      | poor                    | poor                      | moderate (Dirichlet)    | very good        | poor          | good                    | good (if trained) | good (if trained) |
| Filter length      | medium                  | full                      | full                    | minimal (2 taps) | full length   | short                   | full (learned)    | full (learned)    |
| Computational cost | low                     | low (O(N log N))          | low                     | very low         | extremely low | moderate                | high (per patch)  | high (per patch)  |



NOTE: USEFULL FOR PET: all wavelets, PCA / SVD (groupwise) if used poisson-aware, maybe DCT/FFT can be usefull if signaldependent variance is considered (maybe local variance-estimator per coefficient), tensor-decomp may be usefull for nd, but cost intensive (should work if we further also consider poisson-likelihood in optimization)


"""

#######################
### GENERAL IMPORTS ###
#######################

import enum
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pywt
from numpy.typing import NDArray
from scipy.linalg import hadamard
from scipy.ndimage import convolve


class TransformType(enum.Enum):
    DCT = 0
    FFT = 1
    DST = 2
    HAAR = 3
    HADAMARD = 4
    BIOR15 = 5
    BIOR22 = 6
    BIOR33 = 7
    DB1 = 8
    DB2 = 9
    STARLET = 10
    ATROUS = 11
    B3SPLINE = 12
    CURVELET = 13


class TransformMode(enum.Enum):
    ND = 0
    GROUP = 1
    SPATIAL = 2


@dataclass
class Transform:
    type: TransformType
    mode: TransformMode
    parameters: dict[str, Any] = field(
        default_factory=dict
    )  # this is needed for some transforms that need extra parameters


@dataclass
class TransformOperators:
    forward_operator_list: list[NDArray | Callable]
    inverse_operator_list: list[NDArray | Callable]
    is_matrix_operator: bool
    # this is for the phase c transfroms


##########################################
### Transform Matrix creation (Master) ###
##########################################


def get_transform_matrix(
    data_shape: tuple[int, ...], transform: Transform | list[Transform]
) -> tuple[list[NDArray | Callable | None], list[NDArray | Callable | None]]:
    if type(transform) is Transform:
        transform = [transform]

    ndim = len(data_shape)
    forward_transform_list: list[NDArray | Callable | None] = [None] * ndim
    inverse_transform_list: list[NDArray | Callable | None] = [None] * ndim

    assert type(transform) is list
    for T in transform:
        match T.type:
            # Phase A
            case TransformType.DCT:
                forward_matrices, inverse_matrices = init_dct_matrix(data_shape, T.mode)
            case TransformType.FFT:
                forward_matrices, inverse_matrices = init_fft_matrix(data_shape, T.mode)
            case TransformType.DST:
                forward_matrices, inverse_matrices = init_dst_matrix(data_shape, T.mode)
            case TransformType.HAAR:
                if T.mode == TransformMode.GROUP:
                    forward_matrices, inverse_matrices = init_group_transform_matrix(
                        data_shape, T.type
                    )
                else:
                    forward_matrices, inverse_matrices = init_wavelet_matrix(
                        data_shape, "haar", T.mode
                    )
            case TransformType.HADAMARD:
                forward_matrices, inverse_matrices = init_hadamard_matrix(
                    data_shape, T.mode
                )
            # Phase B
            case (
                TransformType.BIOR15
                | TransformType.BIOR22
                | TransformType.BIOR33
                | TransformType.DB1
                | TransformType.DB2
            ):
                wavelet_dict = {
                    TransformType.BIOR15: "bior1.5",
                    TransformType.BIOR22: "bior2.2",
                    TransformType.BIOR33: "bior3.3",
                    TransformType.DB1: "db1",
                    TransformType.DB2: "db2",
                }
                forward_matrices, inverse_matrices = init_wavelet_matrix(
                    data_shape, wavelet_dict[T.type], T.mode
                )
            # Phase C: operator-based transforms (no N x N matrices)
            case (
                TransformType.STARLET
                | TransformType.ATROUS
                | TransformType.B3SPLINE
                | TransformType.CURVELET
            ):
                parameters = {
                    "transform_type": T.type.value,
                    "number_of_levels": getattr(T, "number_of_levels", None),
                    "number_of_scales": getattr(T, "number_of_scales", None),
                    "number_of_angles": getattr(T, "number_of_angles", None),
                    "include_finest_scale": getattr(T, "include_finest_scale", True),
                }

                forward_operator, inverse_operator = init_wavelet_or_curvelet_ops(
                    parameters
                )

                # wrap operators inside lists to remain consistent with matrix-based API
                forward_matrices = [forward_operator]
                inverse_matrices = [inverse_operator]

            # Phase D: unknown transform type
            case _:
                raise ValueError(f"Unknown transform: {T}")

        for i, m in enumerate(forward_matrices):
            if m is not None:
                forward_transform_list[i] = m
                inverse_transform_list[i] = inverse_matrices[i]

    return forward_transform_list, inverse_transform_list


#######################
### APPLY TRANSFORM ###
#######################


def _apply_matrix_along_axis(
    data: NDArray,
    matrix: NDArray,
    axis: int,
) -> NDArray:
    if axis == 0:
        front_shape = data.shape
        reshaped = data.reshape(front_shape[0], -1)
        return (matrix @ reshaped).reshape(front_shape)

    if axis == data.ndim - 1:
        back_shape = data.shape
        reshaped = data.reshape(-1, back_shape[-1])
        return (reshaped @ matrix.T).reshape(back_shape)

    if axis == 1:
        swapped = np.swapaxes(data, 0, 1)
        swapped_shape = swapped.shape
        reshaped = swapped.reshape(swapped_shape[0], -1)
        transformed = (matrix @ reshaped).reshape(swapped_shape)
        return np.swapaxes(transformed, 0, 1)

    moved = np.moveaxis(data, axis, 0)
    moved_shape = moved.shape
    reshaped = moved.reshape(moved_shape[0], -1)
    transformed = (matrix @ reshaped).reshape(moved_shape)
    return np.moveaxis(transformed, 0, axis)


def apply_transform_nd(
    data: NDArray, transform_list: list[NDArray | Callable | None], forward: bool = True
) -> NDArray:
    # data: ndarray, any dimension
    out = data
    for axis, T in enumerate(transform_list):
        # skip axis if no transform given
        if T is None:
            continue

        # Operator-based transform: T must already
        # be the correct forward/inverse operator
        if callable(T):
            out = T(out.copy())
            continue

        out = _apply_matrix_along_axis(out, T, axis)
    return out


#################
### TRANFORMS ###
#################


def generate_matrix_lists(
    data_shape: tuple[int, ...],
    mode: TransformMode,
    forward_generator: Callable,
    inverse_generator: Callable,
) -> tuple[list[NDArray], list[NDArray]]:
    axis_list = list(data_shape)
    match mode:
        case TransformMode.ND:
            pass
        case TransformMode.GROUP:
            axis_list[1:] = [0] * (len(axis_list) - 1)
        case TransformMode.SPATIAL:
            axis_list[0] = 0
        case _:
            raise ValueError(f"Mode {mode} not available.")

    forward_list = []
    inverse_list = []
    for axis in axis_list:
        if axis == 0:
            forward_list.append(None)
            inverse_list.append(None)
        else:
            fwm = forward_generator(axis)
            if type(fwm) is tuple:
                forward_list.append(fwm[0])
                inverse_list.append(fwm[1])
                continue
            invm = inverse_generator(axis)
            forward_list.append(fwm)
            inverse_list.append(invm)
    return forward_list, inverse_list


###############
### PHASE A ###
###############


### DCT, FFT and DST ###


### DCT (Dsicrete Cosine Transform) ###

r""" 

The DCT takes a signal (list of array of numbers) and rewrites 
it as a sum of cosine waves with different frequencies and 
amplitudes.
It does not lose information. We can transform back exactly, 
but it changes how the information is represented.


Intuition:
- We got a signal x[n] of length N.
- DCT does not store this as individual raw values but rather 
  describes how much of each cosine pattern is needed to rebuild 
  that signal
- Low-freq cosine terms describe smooth, slowly changing parts.
- High-frq terms describe sharp edges or noise.
- In most natural signals, most of the energy is in the first 
  few (low-freq) coefficients, i.e. this allows compression or 
  denoising by ignoring small high-frequency coefficients.

  
Mathematical definition in Overleaf document!


Needed Inputs to compute a DCT:
n              : Length of the dimension (number of samples per axis)
type           : DCT type (usually type-II for forward, type-III for inverse)
norm           : Normalization convention (usually 'ortho' for othonorma basis)
Optional' axis : for multidimensional data, i.e. which axis the transforms acts on

Note           : DCT tyeps - there are four standardised variants. They divert only 
                 in their boarder behavior. This changes how the cosine-wave is mirrored or shifted.
                 We will not do a deep dive onto the four types. A quick overview though:
                 type1 - typically only included for mathematical completion, not relevant
                 type2 - standard-dct (used for bm3d/bm4d) - saves energy, represents orthonormally
                 type3 - inverse of type2 - precise reconstruction of the date
                 type4 - selfinvers (used for audio-codes (MDCT)) - if symmetric boarders are needed

Note           : Normalization   - for this we only consider two options to be chosen from:
                 1. norm = None  - no normalization at all. This leads to no scaling -> the sum can distore the energy
                 2. norm = ortho - orthonomalization. Scales it such that the DCT is orthonormal (duhh) -> scales the first comp with \sqrt(1/N) and any further iwth \sqrt(2/N)

                 
"""


### Initialization DCT ###

# forward standard dct transform (type ii)


def init_dct_matrix(
    data_shape: tuple[int, ...], mode: TransformMode
) -> tuple[list[NDArray], list[NDArray]]:
    def dct_matrix(size: int) -> NDArray:
        dct_mat = np.zeros((size, size), dtype=np.float32)
        for frequency_index in range(size):
            for sample_index in range(size):
                normalization = (
                    np.sqrt(1 / size) if frequency_index == 0 else np.sqrt(2 / size)
                )
                dct_mat[frequency_index, sample_index] = normalization * np.cos(
                    np.pi * (sample_index + 0.5) * frequency_index / size
                )
        return dct_mat

    def idct_matrix(size: int) -> NDArray:
        idct_mat = np.zeros((size, size), dtype=np.float32)
        for sample_index in range(size):
            for frequency_index in range(size):
                normalization = (
                    np.sqrt(1 / size) if frequency_index == 0 else np.sqrt(2 / size)
                )
                idct_mat[sample_index, frequency_index] = normalization * np.cos(
                    np.pi * (sample_index + 0.5) * frequency_index / size
                )
        return idct_mat

    return generate_matrix_lists(data_shape, mode, dct_matrix, idct_matrix)


def _is_power_of_two(size: int) -> bool:
    return size > 0 and (size & (size - 1)) == 0


def _hierarchical_haar_matrix(size: int) -> NDArray[np.float32]:
    if not _is_power_of_two(size):
        raise ValueError(f"Haar group size must be a power of two, got {size}.")

    matrix = np.ones((1, 1), dtype=np.float32)
    scale = np.float32(1.0 / np.sqrt(2.0))
    current_size = 1
    while current_size < size:
        coarse = np.kron(matrix, np.array([scale, scale], dtype=np.float32))
        detail = np.kron(
            np.eye(current_size, dtype=np.float32),
            np.array([scale, -scale], dtype=np.float32),
        )
        matrix = np.vstack((coarse, detail)).astype(np.float32, copy=False)
        current_size *= 2
    return matrix


def init_group_transform_matrix(
    data_shape: tuple[int, ...], transform_type: TransformType
) -> tuple[list[NDArray | None], list[NDArray | None]]:
    if transform_type != TransformType.HAAR:
        raise ValueError(f"Unsupported group transform type: {transform_type}")

    group_size = data_shape[0]
    forward_list: list[NDArray | None] = [None] * len(data_shape)
    inverse_list: list[NDArray | None] = [None] * len(data_shape)

    if _is_power_of_two(group_size):
        forward_matrix = _hierarchical_haar_matrix(group_size)
        inverse_matrix = forward_matrix.T.copy()
    else:
        forward_matrix = np.zeros((group_size, group_size), dtype=np.float32)
        for frequency_index in range(group_size):
            normalization = (
                np.sqrt(1 / group_size)
                if frequency_index == 0
                else np.sqrt(2 / group_size)
            )
            for sample_index in range(group_size):
                forward_matrix[frequency_index, sample_index] = normalization * np.cos(
                    np.pi * (sample_index + 0.5) * frequency_index / group_size
                )
        inverse_matrix = forward_matrix.T

    forward_list[0] = forward_matrix
    inverse_list[0] = inverse_matrix
    return forward_list, inverse_list


# FFT #


"""
FFT (Fast Fourier Transform) is an algorithm for the effectiv and quick implementation of a DFT (discrete fourier transform).
It represents signals in the freq domain using complex basis functions combined of sin and cosin.

Properties:
- global, not localized in space
- orthogonal (unitary in cmplx domain)
- perfect energy preservation (parseval)
- efficient O(N log N) computation (because of FFT)
- bad on edges and local variations 
- good for periodic /freq-sparse signal


"""


def init_fft_matrix(
    data_shape: tuple[int, ...], mode: TransformMode
) -> tuple[list[NDArray], list[NDArray]]:
    def fft_matrix(size: int) -> NDArray:
        n = np.arange(size)
        k = n.reshape((size, 1))
        return np.exp(-2j * np.pi * k * n / size) / np.sqrt(size)

    def ifft_matrix(size: int) -> NDArray:
        n = np.arange(size)
        k = n.reshape((size, 1))
        return np.exp(2j * np.pi * k * n / size) / np.sqrt(size)

    return generate_matrix_lists(data_shape, mode, fft_matrix, ifft_matrix)


# DST #


"""
DST (discrete sine transf) is a real valued orthogonal transf.
Basically DFT but with online sine func.
Assumes the signal is odd-symmetric at the boundaries, reducing edge artifacts.

Properties:
- global, real, orthogonal
- better at edges than DFT
- good energy compaction for smooth and oscillating signals
- well suited for diffusion-like or dirichlet boundary problems
"""


def init_dst_matrix(
    data_shape: tuple[int, ...], mode: TransformMode
) -> tuple[list[NDArray], list[NDArray]]:
    def dst_matrix(size: int) -> NDArray:
        n = np.arange(size, dtype=np.float32)
        k = n.reshape((size, 1))
        return np.asarray(
            np.sqrt(2 / (size + 1)) * np.sin(np.pi * (k + 1) * (n + 1) / (size + 1)),
            dtype=np.float32,
        )

    def idst_matrix(size: int) -> NDArray:
        n = np.arange(size, dtype=np.float32)
        k = n.reshape((size, 1))
        return np.asarray(
            np.sqrt(2 / (size + 1)) * np.sin(np.pi * (n + 1) * (k + 1) / (size + 1)),
            dtype=np.float32,
        )

    return generate_matrix_lists(data_shape, mode, dst_matrix, idst_matrix)


### Hadamard Transform ###

"""
The Hadamard transform is an orthogonal, purely binary transform.
It only uses +-1 as coeffs.
It decomposes the signal into a seq. of additive and subtractive patterns,
similar to DCT but w/o functions.

Idea:
- Each transform vector consists of +-1
- Orthogonal (i.e. inverse = transpose)
- capture correlations through additive combinations, not freq. 

Properties:
- orthogonal and self-inverse
- no multiplications (only additions/subtractions)
- extremely fast and numerically stable
- performs best on structured, low-frequency-dominant data
- poor localization, i.e. weak on highly non-stationary signals



Why Hadamard:
- good for simple, fast noise-robust transforms.
- numerical stable and orthogonal -> simplifies PURE-LET derivations
- good when signal energy is broadly distr. (i.e. no sharp peaks)
- good test / baseline

Issues:
- globle -> poor local structures and edges
- worse than DCT with smooth images and worse than wavelet for localized textures

Summary: 
simple, quick, easy, but not the best for most. Good for computation and test, probably never optimal.
"""


### Initialization Hadamard ###


def init_hadamard_matrix(
    data_shape: tuple[int, ...], mode: TransformMode
) -> tuple[list[NDArray], list[NDArray]]:
    def hadamard_matrix(size: int) -> NDArray:
        if not (size and ((size & (size - 1)) == 0)):
            raise ValueError(f"Hadamard matrix size {size} must be a power of two.")
        base = hadamard(size).astype(np.float32)
        return base / np.float32(np.sqrt(size))

    return generate_matrix_lists(data_shape, mode, hadamard_matrix, hadamard_matrix)


### WAVELETS GENERAL ###
### WAVELETS GENERAL ###
### WAVELETS GENERAL ###


### bior1.5 ###

"""
Wavelet is a type of transform, that disects the signal into different 
frequency areas, similar to a DCT or an FFT, though with higher 
resolution in time/space.

Instead of analyzing the entire signal, using Sine and Cosine waves, 
Wavelets uses small, scaled, shifted waveforms (Wavelets).
The result: A representation, that is localised in Space (Pixelspace) and Frequency (Details).

Thus Wavelt transforms provide:
- Approximate coefficients (low freq, rough structures)
- detail coeffs (high freq, edges, textures)


Biorthogonal-Wavelets:
- there are two different wavelet pairs:
    - One pair for forward transformation (analysis)
    - One pair for the backwards transformation (synthesis)
- the two pair are non-identical, though biorthogonal towards oneanother,
    i.e. the exact reconstruction remains possible.

This allows for wavelets to be symmetric or compactly supported,
    i.e. wavelets may have properties that are useful for real images.


    
Meaning of bior1.5:
"bior1.5" reffers to a special family (Cohen-Daubechies-Feauveau) of biorthogonal wavelets.
The 1.5 reffers to 1 -- number of vanishing moments of the reconstruction (synthesis) filter
and 5                -- number of vanishing moments of the decomposition (analysis) filter
A vanishing moment means that the wavelet can represent low-order polynomials smoothly, 
affecting smoothness and approximation accuracy.

Concretely:
- bior1.5 uses symmetric filters (no phase distortion)
- short filter length -> low computational cost
- strong energy compaction for smooth, natural images
- perfect reconstruction due to biorthogonal filter pairs




why bior1.5 for denoising // hard thresholding in particular:
When applying hard thresholding we set small coefficients (most likely noise)
to 0, while large ones (structures) are kept.
For this to work, we need a transformation basis, such that the signal energy
gets concentrated, while the noise energy remains spread.
This makes bior1.5 an ideal candidate:
- high energy compression -> real structures will appear in fewer, though larger coefficients
- local basefuncition     -> noise continues to be diffuse
- symmetric filter        -> no edge-artifacts #

Therefore, bior1.5 is typically used for Hard Thresholding.
It is less suitable for Wiener filtering, since Wiener filtering assumes 
a linear frequency-domain representation (where DCT or FFT are superior). 


NOTE: Also consider bior2.2, bior3.3, db1, db2, or Haar for specific applications.

"""


### Haar ###

r"""

The Haar wavelet is the most simple orthonormal wavelet transform.
It decomposes the signal into pairs of averages and differences.
Therefore it represents overall trend (low freq) as well as abrupt
local changes (high freq).

Idea:
- Each pair of adjacent samples is replaced by:
    - one average    ( approx coeff)
    - one difference ( detail coeff)

mathematically: simple orthogonal matrix with +-1 entries and scaled by 1/\sqrt{2}.
NOTE: this is effectively similar to Hadamard though localized.

Properties: 
- orthogonal
- self inverse (transpose = inverse)
- compact support ( each wavelet effects only a few samples)
- fast computation
- captures sharp edges and discontinuity very well
- bad for repr smooth or curved structures




Why Haar for denoising:
- Simple and robust baseline for wavelet-domain methods.
- Very fast, low-memory, and easy to extend to N dimensions.
- Works well for signals or patches that are piecewise constant, 
  i.e. with sharp transitions and few intensity levels.
- Often used as a first test for thresholding or PURE-LET formulations, 
  since its orthogonality simplifies analytical derivations.

However:
- Haar performs poorly for smooth or textured regions, 
  since it lacks frequency selectivity.
- In high-dimensional denoising (e.g. PET sinograms), 
  it can still serve as a first-stage coarse filter, 
  or as a preconditioning step before applying more complex transforms (e.g. bior or DCT).

NOTE:
Haar is ideal when speed and locality matter more than precision.
It is orthogonal, separable, fast, and analytically simple, 
making it an excellent starting point for N-D thresholding pipelines.

"""


### Initialization Wavelet ###

# This isnitialization supports the following (and further, not listed) wavelets:
## orthogonal wavelets   --- haar, db1-db10, [sym2-sym8, coif1-coif5] [only haar has been studied yet]
## biorthogonal wavelets --- bior1.3, bior1.5, bior2.2, [...]
## reverse biorthogonal  --- [...]
## CDF                   --- bior & rbio families
## Symlets / Daubechies  --- [...]
## [...]


def init_wavelet_matrix(
    data_shape: tuple[int, ...], wavelet_name: str, mode: TransformMode
) -> tuple[list[NDArray], list[NDArray]]:
    wavelet = pywt.Wavelet(wavelet_name)
    is_orthogonal = wavelet.orthogonal

    def get_official_bior15_matrix(
        signal_length: int,
    ) -> tuple[NDArray, NDArray] | None:
        if wavelet_name != "bior1.5":
            return None

        if signal_length == 4:
            forward_matrix = np.array(
                [
                    [0.50000000, 0.50000000, 0.50000000, 0.50000000],
                    [0.50000000, 0.50000000, -0.50000000, -0.50000000],
                    [0.70710677, -0.70710677, 0.0, 0.0],
                    [0.0, 0.0, 0.70710677, -0.70710677],
                ],
                dtype=np.float32,
            )
        elif signal_length == 8:
            forward_matrix = np.array(
                [
                    [
                        0.343550200747110,
                        0.343550200747110,
                        0.343550200747110,
                        0.343550200747110,
                        0.343550200747110,
                        0.343550200747110,
                        0.343550200747110,
                        0.343550200747110,
                    ],
                    [
                        -0.225454819240296,
                        -0.461645582253923,
                        -0.461645582253923,
                        -0.225454819240296,
                        0.225454819240296,
                        0.461645582253923,
                        0.461645582253923,
                        0.225454819240296,
                    ],
                    [
                        0.569359398342840,
                        0.402347308162280,
                        -0.402347308162280,
                        -0.569359398342840,
                        -0.083506045090280,
                        0.083506045090280,
                        -0.083506045090280,
                        0.083506045090280,
                    ],
                    [
                        -0.083506045090280,
                        0.083506045090280,
                        -0.083506045090280,
                        0.083506045090280,
                        0.569359398342840,
                        0.402347308162280,
                        -0.402347308162280,
                        -0.569359398342840,
                    ],
                    [
                        0.707106781186550,
                        -0.707106781186550,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    [
                        0.0,
                        0.0,
                        0.707106781186550,
                        -0.707106781186550,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.707106781186550,
                        -0.707106781186550,
                        0.0,
                        0.0,
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.707106781186550,
                        -0.707106781186550,
                    ],
                ],
                dtype=np.float32,
            )
        else:
            return None

        inverse_matrix = np.asarray(np.linalg.inv(forward_matrix), dtype=np.float32)
        return forward_matrix, inverse_matrix

    def build_wavelet_matrix(
        signal_length: int,  # , wavelet: pywt.Wavelet
    ) -> tuple[NDArray, NDArray]:
        official_matrix = get_official_bior15_matrix(signal_length)
        if official_matrix is not None:
            return official_matrix

        max_level = pywt.dwt_max_level(
            data_len=signal_length, filter_len=len(wavelet.dec_lo)
        )
        level = min(1, max_level)  # one level sufficient for matrix form

        forward_matrix = np.zeros((signal_length, signal_length), dtype=np.float32)
        inverse_matrix = np.zeros((signal_length, signal_length), dtype=np.float32)

        for i in range(signal_length):
            basic_vectors = np.zeros(signal_length, dtype=np.float32)
            basic_vectors[i] = 1.0

            # Forward transform (analytis)
            coefficients = pywt.wavedec(
                basic_vectors, wavelet, level=level, mode="periodization"
            )
            coefficient_vector = np.concatenate(coefficients)
            forward_matrix[:, i] = np.asarray(
                coefficient_vector[:signal_length],
                dtype=np.float32,
            )

        # Inverse transform (synthesis)
        if is_orthogonal:
            # for orthogonal wavelets: inverse is trivial
            inverse_matrix = forward_matrix.T.astype(np.float32, copy=False)
        else:
            # for biorthogonal wavelets: compute robust inverse
            inverse_matrix = np.asarray(
                np.linalg.pinv(forward_matrix), dtype=np.float32
            )

        return forward_matrix, inverse_matrix

    return generate_matrix_lists(
        data_shape, mode, build_wavelet_matrix, build_wavelet_matrix
    )


### WAVELET GENERAL END ###
### WAVELET GENERAL END ###
### WAVELET GENERAL END ###


###############
### PHASE B ###
###############


### SVD (Singular Value Decomposition) ###

r"""
The singular value decomp (SVD) is a data-dependent linear transform
It expresses a signal or image as a product of three matrices:

M = U \Sigma V^t

where:
    - U contains orthonormal left singular vectors (basis of the column space)
    - \Sigma is a diagonal matrix of singular values (strength of each component)
    - V contains orthonormal right singular vectors (basis of the row space)

Each singular value represents the importance of its correspondijng mode.
Large singular values -> caputre domain structures (e.g. smooth regions or main textures)
small singular values -> represent noise 

Properties:
- fully data dependent 
- optimal low-rank approx 
- good energy compaction
- orthogonal components -> invertible



Comparison

| Property                | DCT                         | bior1.5                    | Haar             | Hadamard        | SVD                 |
|-------------------------|-----------------------------|----------------------------|------------------|-----------------|--------------------|
| Support domain          | global                      | local                      | local (minimal)  | global          | adaptive / local   |
| Symmetry                | semi-symmetric              | symmetric                  | symmetric        | symmetric       | none (data-driven) |
| Orthogonality           | orthogonal                  | biorthogonal               | orthogonal       | orthogonal      | orthogonal (U,V)   |
| Energy compaction       | strong (smooth signals)     | strong (natural images)    | weak (piecewise) | moderate        | very strong        |
| Edge handling           | poor                        | good                       | very good        | poor            | excellent (learned)|
| Filter length           | medium                      | short                      | minimal (2 taps) | full length     | full (data-based)  |
| Computational cost      | low                         | moderate                   | very low         | extremely low   | high               |


Why SVD:
- Concentrates signal energy in a few dominant singular values
- Noise tends to appear in small singular values -> simple thresholding or weighting cleans it
- Allows local adaptiveness, when applied per block or patch group
- good baseline for low-rank modeling methods like PURE-LET (maybe WNNM) low rank models

Limitations:
- Heavy computation (O(N^3)) for large matrices
- Basis not fixed -> requires recomputation for each new block
- Harder to generalize across dimensions; often applied per 2D/3D block or group

NOTE:
SVD gives mathematically clean & adaptive basis for each data block
-> ideal for low rank correlated structures (like our patch stacks)
Energy compaction and denoising strenght come at high computation cost

"""

### Initialization SVD ###

# Prior to SVD we need to flatten the nD Stack


def init_pca_matrix(
    data: NDArray,
    mode: TransformMode = TransformMode.ND,
    n_components: int | None = None,
    center: bool = True,
) -> tuple[list[NDArray], list[NDArray]]:
    if mode is TransformMode.ND:
        shape = data.shape
        ndim = data.ndim - 1  # exclude block dimension
        forward_list = []
        inverse_list = []

        for axis in range(ndim):
            L = shape[axis]
            # move axis to end, keep block dimension last
            perm = [i for i in range(ndim) if i != axis] + [axis, ndim]
            X = np.transpose(data, axes=perm).reshape(-1, L)

            if center:
                mean = X.mean(axis=0, keepdims=True)
                X -= mean

            U, S, Vt = np.linalg.svd(X, full_matrices=False)
            r = L if n_components is None else min(n_components, Vt.shape[0])
            V_reduced = Vt[:r, :]
            Vinv = V_reduced.T

            forward_list.append(V_reduced)
            inverse_list.append(Vinv)

        return forward_list, inverse_list

    elif mode is TransformMode.GROUP:
        Lg = data.shape[-1]
        X = data.reshape(-1, Lg)

        if center:
            mean_g = X.mean(axis=0, keepdims=True)
            X -= mean_g

        Ug, Sg, Vtg = np.linalg.svd(X, full_matrices=False)
        r = Lg if n_components is None else min(n_components, Vtg.shape[0])
        V_reduced = Vtg[:r, :]
        Vinv = V_reduced.T

        return V_reduced, Vinv

    else:
        raise ValueError("mode must be TransformMode.ND or TransformMode.GROUP")


#######################################
### Phase C - non-matrix-transforms ###
#######################################


def init_wavelet_or_curvelet_ops(
    parameters: dict[str, Any],
) -> tuple[Callable, Callable]:

    transform_type = parameters["transform_type"].lower()

    ##########################################################
    ### STARLET  (Isotropic Undecimated Wavelet Transform) ###
    ##########################################################

    if transform_type == "starlet":
        number_of_levels = parameters.get("number_of_levels", 4)

        # Standard B3-spline starlet filter
        starlet_filter = np.array([1, 4, 6, 4, 1], dtype=float) / 16.0

        def starlet_forward(data_array: NDArray):
            current_array = data_array.copy()
            detail_coefficients_list = []

            for level in range(number_of_levels):
                smoothed_array = convolve(current_array, starlet_filter, mode="same")
                detail_coefficients_list.append(current_array - smoothed_array)
                current_array = smoothed_array

            return detail_coefficients_list, current_array

        def starlet_inverse(coefficients):
            detail_coefficients_list, coarse_scale = coefficients
            reconstructed_array = coarse_scale.copy()

            for details in reversed(detail_coefficients_list):
                reconstructed_array += details

            return reconstructed_array

        return starlet_forward, starlet_inverse

    ################################################
    ### À TROUS  (Undecimated Wavelet Transform) ###
    ################################################

    if transform_type == "atrous":
        number_of_levels = parameters.get("number_of_levels", 4)
        atrous_filter = np.array([1, 4, 6, 4, 1], dtype=float) / 16.0

        def atrous_forward(data_array: NDArray):
            current_array = data_array.copy()
            detail_coefficients_list = []

            for level in range(number_of_levels):
                step = 2**level
                upsampled_filter = np.zeros(step * (len(atrous_filter) - 1) + 1)
                upsampled_filter[::step] = atrous_filter

                smoothed_array = convolve(current_array, upsampled_filter, mode="wrap")
                detail_coefficients_list.append(current_array - smoothed_array)
                current_array = smoothed_array

            return detail_coefficients_list, current_array

        def atrous_inverse(coefficients):
            detail_coefficients_list, coarse_scale = coefficients
            reconstructed_array = coarse_scale.copy()
            for details in reversed(detail_coefficients_list):
                reconstructed_array += details
            return reconstructed_array

        return atrous_forward, atrous_inverse

    ###########################
    ### B3-SPLINE TRANSFORM ###
    ###########################

    if transform_type == "b3spline":
        number_of_levels = parameters.get("number_of_levels", 4)
        b3_filter = np.array([1, 4, 6, 4, 1], dtype=float) / 16.0

        def b3spline_forward(data_array: NDArray):
            current_array = data_array.copy()
            detail_coefficients_list = []

            for level in range(number_of_levels):
                step = 2**level
                upsampled_filter = np.zeros(step * (len(b3_filter) - 1) + 1)
                upsampled_filter[::step] = b3_filter

                smoothed_array = convolve(current_array, upsampled_filter, mode="wrap")
                detail_coefficients_list.append(current_array - smoothed_array)
                current_array = smoothed_array

            return detail_coefficients_list, current_array

        def b3spline_inverse(coefficients):
            detail_coefficients_list, coarse_scale = coefficients
            reconstructed_array = coarse_scale.copy()
            for details in reversed(detail_coefficients_list):
                reconstructed_array += details
            return reconstructed_array

        return b3spline_forward, b3spline_inverse

    ##########################
    ### CURVELET TRANSFORM ###
    ##########################

    if transform_type == "curvelet":
        from fdct2 import FDCT2  # oder deine korrekte Import-Location

        fdct_object = FDCT2(
            nscales=parameters.get("number_of_scales", 5),
            nangles=parameters.get("number_of_angles", 16),
            finest=parameters.get("include_finest_scale", True),
        )

        def curvelet_forward(data_array: NDArray):
            return fdct_object.fwd(data_array)

        def curvelet_inverse(curvelet_coefficients):
            return fdct_object.inv(curvelet_coefficients)

        return curvelet_forward, curvelet_inverse

    ####################
    ### Falscher Typ ###
    ####################

    raise ValueError(f"Unknown transform type: {transform_type}")


### PURE-LET DISCUSSION ###

# NOTE: PURE-LET is a denoiser. This is not per say a transform. As this is a new denoiser
#       not native to bm4d though we namend the category after the denoiser, not the
#       transforms, as the first thing that we do, will be discussing the correct transform to use.

# NOTE: Most of the transforms that seem usefull for PURE-LET will be already implemented for
#       other reasons, thus this part will be more about the discussion of which transforms to
#       use rather than implementing a new transform. For those implementations we regard to the
#       certain paragraphs in the code.


# LET - lineare expansion of thresholds

# finds optimal process for restoring the image
# first Wiener filter
# then  transform domain thresholding (detail extracter)
#

## Upsides:
#
# non iterative -> no ill posed problem
# parameter free -> feed image and spits out restored image
#
#
#

# Question: How do we get around the issue of trying to get the optimal MSE?
#
#  - Poisson unbiased risk estimate (PURE)
#  - depends only on observed noisy image
#  - idea: minimize PURE function and you are golden

"""

There is a multitude of transforms that may be usefull for PURE-LET. 
Unlike for Wiener Filtering and Hard-Thresholding, we do not have a single
established transform to use, thus we must evaluate which transform 
provides the best characteristics for our specific data.


Each elementary function consists of a Wiener
filtering followed by a pointwise thresholding of undecimated
Haar wavelet coefficients.

Initial assessment of candidate transforms for PURE-LET (and general BMnD use):

1. DCT (Type-II, orthonormal)
    - Why  : strong energy compaction for smooth or slowly varying components.
             Orthogonal -> inverse = transpose.
             Simple least-squares formulation for PURE-LET risk estimation.
    - When : global / subband-linked coefficients; Wiener / PURE-LET hybrid.
    - Notes: separable -> apply per axis for N-dimensional data.

2. Haar / small orthonormal wavelets (e.g. db1, db2)
    - Why  : highly localized; compact support.
             Effective with sharp edges and sparse signals.
             Orthonormality simplifies subband-based noise modeling.
    - When : very sparse or piecewise constant structures (e.g. PET sinogram patches).
    - Notes: good baseline and fast to compute.
            
3. Biorthogonal wavelets (e.g. bior1.5)
    - Why  : symmetric filters -> low boundary artifacts.
             Excellent for spatially localized structures and hard thresholding.
             Not orthogonal -> need explicit inverse transform.
    - When : images or sinograms with edges or gradual transitions.
    - Notes: inverse reconstruction handled by explicit synthesis filters.
         
4. Hadamard Transform
    - Why  : simple orthogonal +-1 matrix -> very fast via FWHT.
             Strong decorrelation for binary or structured data.
             Energy spread uniform; can reveal structure independent of amplitude.
    - When : low-SNR data, binary patterns, or fast approximate decorrelation.
    - Notes: no floating-point multiplications; purely additions/subtractions.

5. Identity (no transform)
    - Why  : baseline comparison.
             Sometimes PURE-LET or Poisson-LET performs well directly in data domain.
    - When : extremely sparse or low-count Poisson data.
    - Notes: preserves interpretability; useful sanity check.


Notes:
- Orthogonal transforms (DCT, Haar, Hadamard, PCA, SVD) -> easy PURE-LET formulation since inverse = transpose.
- Biorthogonal wavelets -> slightly trickier risk modeling but offer spatial advantages.
- Tensor decomposition -> most general and scalable to 5D+ PET sinogram structures.
- Choice depends on structure: smooth vs. edge vs. low-rank vs. multidimensional.

"""
