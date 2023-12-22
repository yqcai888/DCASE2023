# DCASE 2023 Challenge Task1 Submission - XJTLU
This repository contains the source code of our new ideas for the [DCASE 2023 Challenge Task1](https://dcase.community/challenge2023/task-low-complexity-acoustic-scene-classification). Please refer to our [technical report](https://dcase.community/documents/challenge2023/technical_reports/DCASE2023_Cai_74_t1.pdf) for details.
## tf-sepnet.py
The tf-sepnet.py file contains the implementation of our proposed neural network architecture called **TF-SepNet**. The latest paper has been accepted by ICASSP 2024 and available at [arXiv](https://arxiv.org/abs/2309.08200).
## device_simulate.py
The device_simulate.py file is designed for simulating an audio recording from the original device to new devices. It utilizes impulse response files from the [MicIRP](http://micirp.blogspot.com/?m=1) dataset to synthesize the acoustic characteristics of different devices to the original audio recording.
## common.py
The **Adaptive Residual Normaliztion** has been included in common.py.
## Citation
If you find our code helps, we would appreciate using the following citations:
```
@article{cai2023tf,
  title={TF-SepNet: An Efficient 1D Kernel Design in CNNs for Low-Complexity Acoustic Scene Classification},
  author={Cai, Yiqiang and Zhang, Peihong and Li, Shengchen},
  journal={arXiv preprint arXiv:2309.08200},
  year={2023}
}
```
```
@techreport{Cai2023a,
    Author = "Cai, Yiqiang and Lin, Minyu and Zhu, Chenyang and Li, Shengchen and Shao, Xi",
    title = "{DCASE}2023 Task1 Submission: Device Simulation and Time-Frequency Separable Convolution for Acoustic Scene Classification",
    institution = "Detection and Classification of Acoustic Scenes and Events (DCASE) Challenge",
    year = "2023",
}
```
