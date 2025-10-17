# DPMambaIR

DPMambaIR: All-in-One Image Restoration via Degradaion-Aware Prompt State Space Model

## Installation

See [INSTALL.md](Install.md) for the installation of dependencies required to run Restormer.

## Dataset
Google drive下载路径
[All-in-One数据集下载链接](https://drive.google.com/file/d/11eeZptw6GVAU8EMlw7i7YRH8jxEbPjqA/view?usp=drive_link)

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
预训练降质提取器：[paper report训练权重，非并行结构](https://drive.google.com/file/d/1_VS63VTt-3asDGxK6RKXeiKn-rTK9elx/view?usp=drive_link)

DPMambaIR:[paper report训练权重](https://drive.google.com/file/d/1zEVtk0d69zpsNiN5z-hgWr5v3m__5AeG/view?usp=drive_link)

DPMambaIR-L:[paper report训练权重](https://drive.google.com/file/d/1CPsnzPAhb5XuBrxwZa8WFgzdzOxXTYGp/view?usp=drive_link)
