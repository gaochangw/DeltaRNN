# DeltaRNN
This repo mainly contains:
- Training code of DeltaGRU & DeltaLSTM using the [Google Speech Command Dataset V2](https://arxiv.org/abs/1804.03209)

# Project Structure
```
.
└── config                 # Configuration Files (for feature extractor or whatever else you like).
└── data                   # Dataset Description Files (some are generated after the 'prepare' step).
└── feat                   # Extracted Features (Will be generated after the 'feature' step).
└── log                    # Experiment Log Data (Will be generated after the 'train' step).
└── modules                # PyTorch Modules.
└── networks               # PyTorch Neural Networks.
    ├── layers             # Customized Layers (incl. DeltaGRU, DeltaLSTM).
    ├── models             # Top-Level Network Models.
└── steps                  # Training steps (1 - prepare, 2 - feature, 3 - pretrain, 4 - retrain).
└── utils                  # Libraries of useful methods.
    ├── feature            # Libraries for feature extraction.
└── project.py             # A class defining all major training functions and stores hyperparameters
└── main.py                # Top

```

# Prerequisite
This project was tested with PyTorch 1.13 in Ubuntu 22.04 LTS.

Install Miniconda
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
```

Create an environment and install required packages:
```
conda create -n pt python=3.10 numpy matplotlib pandas scipy tqdm \
    pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

Activate the environment.
```
conda activate pt
```

Install other required packages with pip:
```
pip install soundfile
```

#  DeltaRNN Training
DeltaRNNs can be trained from scratch (randomly initialized parameters) or by following a pretrain(GRU/LSTM)-retrain(DeltaGRU/DeltaLSTM) scheme.
Example:
```
python main.py --dataset gscdv2 --step prepare
python main.py --dataset gscdv2 --step feature
python main.py --dataset gscdv2 --step pretrain --epochs_pretrain 20 --batch_size 32
python main.py --dataset gscdv2 --step retrain  --epochs_retrain 10 --thx 0.1 --thh 0.1 --batch_size 64
```

#  Reference
If you find this repository helpful, please cite our work.
- [TNNLS 2022] Spartus: A 9.4 TOp/s FPGA-Based LSTM Accelerator Exploiting Spatio-Temporal Sparsity
```
@ARTICLE{Gao2022Spartus,
  author={Gao, Chang and Delbruck, Tobi and Liu, Shih-Chii},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={Spartus: A 9.4 TOp/s FPGA-Based LSTM Accelerator Exploiting Spatio-Temporal Sparsity}, 
  year={2022},
  volume={},
  number={},
  pages={1-15},
  doi={10.1109/TNNLS.2022.3180209}}
```
- [FPGA 2018] DeltaRNN: A Power-Efficient Recurrent Neural Network Accelerator
```
@inproceedings{Gao2018DeltaRNN,
  author = {Gao, Chang and Neil, Daniel and Ceolini, Enea and Liu, Shih-Chii and Delbruck, Tobi},
  title = {DeltaRNN: A Power-Efficient Recurrent Neural Network Accelerator},
  year = {2018},
  isbn = {9781450356145},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3174243.3174261},
  doi = {10.1145/3174243.3174261},
  abstract = {Recurrent Neural Networks (RNNs) are widely used in speech recognition and natural language processing applications because of their capability to process temporal sequences. Because RNNs are fully connected, they require a large number of weight memory accesses, leading to high power consumption. Recent theory has shown that an RNN delta network update approach can reduce memory access and computes with negligible accuracy loss. This paper describes the implementation of this theoretical approach in a hardware accelerator called "DeltaRNN" (DRNN). The DRNN updates the output of a neuron only when the neuron»s activation changes by more than a delta threshold. It was implemented on a Xilinx Zynq-7100 FPGA. FPGA measurement results from a single-layer RNN of 256 Gated Recurrent Unit (GRU) neurons show that the DRNN achieves 1.2 TOp/s effective throughput and 164 GOp/s/W power efficiency. The delta update leads to a 5.7x speedup compared to a conventional RNN update because of the sparsity created by the DN algorithm and the zero-skipping ability of DRNN.},
  booktitle = {Proceedings of the 2018 ACM/SIGDA International Symposium on Field-Programmable Gate Arrays},
  pages = {21–30},
  numpages = {10},
  keywords = {delta network, fpga, recurrent neural network, gated recurrent unit, hardware accelerator, deep learning},
  location = {Monterey, CALIFORNIA, USA},
  series = {FPGA '18}
}
```
- [JETCAS 2020] EdgeDRNN: Recurrent Neural Network Accelerator for Edge Inference (AICAS 2020 Best Paper)
```
@ARTICLE{Gao2020EdgeDRNN,
  author={Gao, Chang and Rios-Navarro, Antonio and Chen, Xi and Liu, Shih-Chii and Delbruck, Tobi},
  journal={IEEE Journal on Emerging and Selected Topics in Circuits and Systems}, 
  title={EdgeDRNN: Recurrent Neural Network Accelerator for Edge Inference}, 
  year={2020},
  volume={10},
  number={4},
  pages={419-432},
  doi={10.1109/JETCAS.2020.3040300}}

```
