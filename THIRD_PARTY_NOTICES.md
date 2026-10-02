# Third-party notices

## Figures in README.md containing Anti-UAV300 frames

The qualitative figures under `assets/figures/` are diagnostic renderings produced by this
repository's own scripts, and they display frames from the
[Anti-UAV300](https://github.com/ZhaoJ9014/Anti-UAV) dataset (Jiang et al., *Anti-UAV: A
Large-Scale Benchmark for Vision-Based UAV Tracking*, [arXiv:2101.08466](https://arxiv.org/abs/2101.08466)).
Four low-resolution frames from training sequences `20190925_131530_1_2`,
`20190925_200320_1_7`, `20190925_130434_1_4` and `20190925_101846_1_1` are visible, each
annotated with the model output and the dataset's own bounding box.

This is the one place where dataset pixels appear inside the repository; they are included for
illustration of a method and of its failure modes. The upstream project is published under the
MIT license, but a code license is not asserted to be a blanket redistribution authorization for
the recorded imagery. No dataset archive, split, annotation file or bulk imagery is vendored
here, and `docs/datasets.md` remains the access route. If the Anti-UAV300 authors' terms require
it, these four figures are the only assets that need removal.

## MS² additional-source validation

The optional source audit uses the authors' public
[MS² dataset documentation](https://sites.google.com/view/multi-spectral-stereo-dataset)
and metadata from
[UkcheolShin/MS2-MultiSpectralStereoDataset](https://github.com/UkcheolShin/MS2-MultiSpectralStereoDataset).
The dataset website specifies CC BY-NC-SA 3.0. Cite Shin, Park and Kweon,
*Deep Depth Estimation From Thermal Image*, CVPR2023, pp.1043–1053.
Downloaded data, download scripts and inspected upstream loader code remain in
ignored experiment storage; none is vendored or executed as project code. Our
MIT license does not relicense these assets. Review the dataset terms before
redistributing derivatives. See [source audit](docs/registration_ms2_source.md)
for identities, coordinate conventions, limitations and pending qualification.

## Pretrained detector-free matching diagnostic

The diagnostic imports [Kornia](https://github.com/kornia/kornia)'s LoFTR
implementation (Kornia 0.6.5, Apache-2.0) and the official
[XoFTR](https://github.com/OnderT/XoFTR) implementation at revision
`e0fbea431b30be9742effbf5577c90aa8eb938f9` (Apache-2.0).
XoFTR acknowledges its derivation from [LoFTR](https://github.com/zju3dv/LoFTR).
The external repository and checkpoints are stored only in git-ignored
`experiments/external/`; their upstream license files remain in place. This
repository does not redistribute the checkpoints or the METU-VisTIR dataset.
Checkpoint download sources, SHA256 values and the diagnostic-only empty-match
guard are documented in `docs/detector_free_matching.md`. No upstream checkpoint
parameters are changed.

### MINIMA-XoFTR checkpoint comparison

The optional fixed-panel diagnostic uses the official
[MINIMA](https://github.com/LSXI7/MINIMA) XoFTR checkpoint with the same pinned
XoFTR network implementation. MINIMA's repository provides an Apache-2.0 license;
its license and inspected configuration are retained in the ignored external
asset directory. No downloaded model, imagery or upstream source is redistributed
by this repository. Asset source, observed digest, configuration equivalence and
the limits of the checkpoint-swap experiment are recorded in
`docs/registration_minima.md`. A repository code license is not asserted to be a
blanket redistribution authorization for all model training data.

## Frozen DINOv2 feature diagnostic

The optional diagnostic imports the official [DINOv2](https://github.com/facebookresearch/dinov2)
ViT-S/14 model with four register tokens at revision
`7764ea0f912e53c92e82eb78a2a1631e92725fc8`. The DINOv2 code and ordinary DINOv2
weights are distributed under Apache-2.0; the upstream license remains in the
git-ignored external checkout. Other upstream models with different licenses are
not used. Official weights are downloaded from Meta's `dl.fbaipublicfiles.com`,
SHA256 `f433177089a681826f849f194ece3bb48f4d63fb38d32fc837e3dc7a4e5641fb`.
No checkpoint is redistributed in this repository or fine-tuned by this diagnostic.

## Automatic SAM silhouette diagnostic

The optional automatic mask probe imports `segment-anything==1.0`, implementing
[Meta's Segment Anything](https://github.com/facebookresearch/segment-anything)
(Apache-2.0). It downloads the official ViT-B checkpoint from
`https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth` into the
git-ignored `experiments/external/` directory. Installed Python source hashes and
the checkpoint SHA256 are recorded in each report. The first download trusts the
official HTTPS source; no independently published digest is asserted. No SAM
weights or dataset are redistributed, and this diagnostic does not fine-tune SAM.

## Optional DiffV2IR inference integration

The optional sampler imports the official
[DiffV2IR](https://github.com/LidongWang-26/DiffV2IR) code at revision
`495947eaad0c2f3bb380ddf5f8820c2b1443e23f`. Its Apache-2.0 license remains in the
git-ignored external checkout under `experiments/external/DiffV2IR`. This bridge
does not vendor upstream model code or redistribute model weights, CLIP assets
or training datasets. The code license alone does not establish weight/dataset
permissions or absence of benchmark exposure; those checks and a tested runtime
remain pending. See `docs/diffv2ir_integration.md` for scope and preprocessing
differences. CPU test-double outputs are not DiffV2IR experimental results.

## Optional RAFT-Stereo depth diagnostic

The optional MS² comparison imports the official
[RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo) implementation at
`6e93ed2169bd858dbb43033988563f3b0bb49506`. Its MIT license, copyright
2021 Princeton Vision & Learning Lab, is retained in the ignored external
checkout. The authors' linked Middlebury checkpoint has locally observed SHA256
`d22e84c0e431bf31d7cc66902c40601859eb40b35ef7f4399ea81276c2915819`.
Neither upstream code nor weights are redistributed in this repository.
The code license alone is not asserted to establish all training-data or
checkpoint redistribution rights. This diagnostic does not fine-tune the model.
See `docs/registration_ms2_raft_stereo.md` for source, input and evidence limits.

## SuperFusion dense matcher

`src/aero_ir/registration/superfusion.py` adapts the dense registration
architecture from [SuperFusion](https://github.com/Linfeng-Tang/SuperFusion),
revision `bee015a8938bee549d0132b80c1daf161ff80660`. Only the registration network is
included; its hard-coded CUDA device selection and CUDA-only fusion modules are
not included.

Copyright (c) 2022 Linfeng Tang

Licensed under the MIT License:

> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.
