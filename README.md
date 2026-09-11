# Direct Poisson Denoising by N-Dimensional Block-Matching and Collaborative Filtering

BMND is an algorithm for denoising of Poisson corrupted n-dimensional data.

This implementation is based on the paper
```
Christof Duhme, Lars Schiefelbein, Florian Büther and Xiaoyi Jiang,
''Direct Poisson Denoising by N-Dimensional Block-Matching and Collaborative Filtering'',
In: arXiv preprint arxiv:DOI, 2026.
```

The basic algorithms we build upon are BM3D/BM4D, introduced and developed in these publications:
```
Kostadin Dabov, Alessandro Foi, Vladimir Katkovnik and Karen Egiazarian,
''Image Denoising by Sparse 3-D Transform-Domain Collaborative Filtering'',
In: IEEE Transactions on Image Processing, vol. 16, no. 8, pp. 2080-2095, 2007.
DOI: 10.1109/TIP.2007.901238.

Matteo Maggioni, Vladimir Katkovnik, Karen Egiazarian and Alessandro Foi,
''A Nonlocal Transform-Domain Filter for Volumetric Data Denoising and Reconstruction'',
In: IEEE Transactions on Image Processing, vol. 22, no. 1, pp. 119-133, 2013.
DOI: 10.1109/TIP.2012.2210725.

Ymir Mäkinen, Lucio Azzari and Alessandro Foi,
''Collaborative Filtering of Correlated Noise: Exact Transform-Domain Variance for Improved Shrinkage and Patch Matching'',
In: IEEE Transactions on Image Processing, vol. 29, pp. 8339-8354, 2020.
DOI: 10.1109/TIP.2020.3014721.
```

## Usage

```python
import numpy as np
from bmnd import BM3DProfile, NoiseModel, bmnd

# Generate a synthetic 2D image with Poisson noise.
image = np.full((64, 64), 5.0, dtype=np.float32)
image[16:48, 16:48] = 20.0
noisy = np.random.default_rng(0).poisson(image).astype(np.float32)

profile = BM3DProfile(noise_model=NoiseModel.POISSON)
denoised = bmnd(noisy, profile)
```

For your own data, replace `noisy` with a non-negative NumPy array in count
units, without normalizing it to `[0, 1]`. No noise standard deviation is needed
for Poisson denoising. The output has the same shape as the input.

We provide `BM3DProfile` for 2D images and `BM4DProfile` for 3D volumes to match
the original algorithms defaults.
