# DCASE 2023 Challenge Task1 Submission - XJTLU
This repository contains the source code of our new ideas for the [DCASE 2023 Challenge Task1](https://dcase.community/challenge2023/task-low-complexity-acoustic-scene-classification). Please refer to our [technical report](https://dcase.community/documents/challenge2023/technical_reports/DCASE2023_Cai_74_t1.pdf) for details.
## tf-sepnet.py
The tf-sepnet.py file contains the implementation of our proposed neural network architecture called **TF-SepNet**. The latest paper has been published on arXiv. [Link here](https://arxiv.org/abs/2309.08200).
## device_simulate.py
The device_simulate.py file is designed for simulating an audio recording from the original device to new devices. It utilizes impulse response files from the [MicIRP](http://micirp.blogspot.com/?m=1) dataset to synthesize the acoustic characteristics of different devices to the original audio recording.
