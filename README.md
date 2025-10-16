# DPMambaIR

DPMambaIR

## Installation

See [INSTALL.md](Install.md) for the installation of dependencies required to run Restormer.

## Dataset
Google drive下载路径


## Train
注意在yml文件中配置权重路径以及数据集路径

<b> 特别注意预训练降质提取器的权重加载路径在对应的模型下[basicsr/models/arch/DPMambaIR_arch.py第617行，其他版本的模型也类似] </b>
```python
cd basicsr
python train.py -opt options/train/train_XXX.yml

DPMambaIR: basicsr/options/train/train_AIO_DPMambaIR.yml
DPMambaIR-L:  basicsr/options/train/train_AIO_DPMambaIR_Para4.yml
```

## Test
注意在yml文件中配置权重路径以及数据集路径
```python
cd basicsr
python test.py -opt options/test/test_XXX.yml

DPMambaIR: basicsr/options/test/test_DPMambaIR.yml
DPMambaIR-L: basicsr/options/test/test_DPMambaIRnewDE_para4.yml
```


## Pth Install Prepare
Google drive下载路径

DPMambaIR:

DPMambaIR-L:
