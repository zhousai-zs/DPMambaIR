# Installation

1. Clone our repository
```
git clone XXX
cd DPMambaIR
```

2. Make conda environment
```
conda create -n DPMambaIR python=3.10
conda activate DPMambaIR
```

3. Install dependencies
```
install mamba[参考CSDN教程]
install pytorch开发环境
[安装其他依赖]
pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm
pip install einops gdown addict future lmdb numpy pyyaml requests scipy tb-nightly yapf lpips
```

4. Install basicsr
```
python setup.py develop --no_cuda_ext
```

### Download datasets from Google Drive

[All-in-One数据集下载链接](https://drive.google.com/file/d/11eeZptw6GVAU8EMlw7i7YRH8jxEbPjqA/view?usp=drive_link)