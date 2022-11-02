# Adaptive Local Implicit Image Function for Arbitrary-scale Super-resolution

---

This is the official implementation of our paper [Adaptive Local Implicit Image Function for Arbitrary-scale Super-resolution](), accepted by the International Conference on Image Processing (ICIP), 2022. 

## Main Contents

---

### 1. Introduction

- Image representation is critical for many visual tasks. Instead of representing images discretely with 2D arrays of pixels, a recent study, namely local implicit image function (LIIF), denotes images as a continuous function where pixel values are expansion by using the corresponding coordinates as inputs. Due to its continuous nature, LIIF can be adopted for arbitrary-scale image super-resolution tasks, resulting in a single effective and efficient model for various up-scaling factors. However, LIIF often suffers from structural distortions and ringing artifacts around edges, mostly because all pixels share the same model, thus ignoring the local properties of the image. In this paper, we propose a novel adaptive local image function (A-LIIF) to alleviate this problem. Specif- ically, our A-LIIF consists of two main components: an encoder and a expansion network. The former captures cross-scale image features, while the latter models the continuous up-scaling function by a weighted combination of multiple local implicit image functions. Accordingly, our A-LIIF can reconstruct the high-frequency textures and structures more accurately. Experiments on multiple benchmark datasets verify the effectiveness of our method.

### 2. Running the code

**2.1. Preliminaries**

- For `train_liif.py` or `test.py`, use `--gpu [GPU]` to specify the GPUs (e.g. `--gpu 0` or `--gpu 0,1`).
- For `train_liif.py`, by default, the save folder is at `save/_[CONFIG_NAME]`. We can use `--name` to specify a name if needed.

**2.2. DIV2K experiments**

**Train**: `python train_liif.py --config configs/train-div2k/train_edsr-baseline-A-liif.yaml` (with EDSR-baseline backbone).

**Test**: `bash scripts/test-div2k.sh [MODEL_PATH] [GPU]` for div2k validation set, `bash scripts/test-benchmark.sh [MODEL_PATH] [GPU]` for benchmark datasets. `[MODEL_PATH]` is the path to a `.pth` file, we use `epoch-last.pth` in corresponding save folder.

### 4. Results

- Some of the test results can be downloaded. ( [Google Drive](https://drive.google.com/drive/folders/1lympRcAHDVh7wDtXS9bVpWKFYMoPWT7x?usp=sharing) )

### 5. Citation

If our work or this repo is useful for your research, please cite our paper as follows:
```
@inproceedings{li2022adaptive,
  title={Adaptive Local Implicit Image Function for Arbitrary-scale Super-resolution},
  author={Li, Hongwei and Dai, Tao and Li, Yiming and and Zou, Xueyi and Xia, Shu-Tao},
  booktitle={ICIP},
  year={2022}
}
```

### 6. Acknowledge

The code is built on [LIIF](https://github.com/yinboc/liif). We thank the authors for sharing the codes.
