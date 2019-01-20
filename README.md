# An Energy-efficient Real-time Speech Recognition System

This is a project under the ETH Deep Learning course 2018.

## Getting Started

These instructions will guide you to setup your environment and reproduce the results in the report.

### Prerequisites
- System & Hardware
    - Ubuntu 16.04/18.04
    - Single GPU with at least 8 GB VRAM (We used NVIDIA GTX 1080), experiment is unlikely to run with less than 8 GB VRAM
    - Install CUDA 8/9/10 and CuDNN 6/7 (Tested with CUDA 10.0.130 + CuDNN 7.4.1 || CUDA 9.0.176 + CuDNN 7.4.1)

- Environment & Critical Packages
    - Anaconda/Miniconda 64-bit with Python 3.6
    - [PyTorch 1.0.0](https://pytorch.org)
    - [warp-ctc](https://github.com/SeanNaren/warp-ctc)


### Installing

Please follow this section step by step in order to setup the environment and run the code.

- Install Anaconda/Miniconda 64-bit
- Create an conda environment (named pt1) using the provided environment.yml
    ```
    conda env create -f environment.yml
    ```
- Activate the enviroment
    ```
    source activate pt1
    ```
- Install PyTorch 1.0.0 according to your CUDA version
    * with CUDA 8
    ```
    conda install pytorch torchvision cuda80 -c pytorch
    ```
    * with CUDA 9
    ```
    conda install pytorch torchvision -c pytorch
    ```
    * with CUDA 10
    ```
    conda install pytorch torchvision cuda100 -c pytorch
    ```

- Build & Install warp-ctc
    * Clone the repo to your home folder
    ```
    cd ~
    git clone https://github.com/SeanNaren/warp-ctc.git
    ```
    * Build warp-ctc
    ```
    cd warp-ctc
    mkdir build; cd build
    cmake ..
    make -j 8
    ```

    * Install warp-ctc
    ```
    cd ../pytorch_binding
    python setup.py install
    ```

## Running the tests
- Modify path to datasets (TIMIT and QUT-NOISE-TIMIT) in path.sh
- Run run.sh to test the code and get all results
    ```
    ./run.sh
    ```
## Public code used
- [ctc_decoder.py](https://gist.github.com/awni/56369a90d03953e370f3964c826ed4b0) Beam Search by Awni Hannun

## Authors

* **Chang Gao** - *Institute of Neuroinformatics, University of Zurich and ETH Zurich* - [Chang Gao's Personal Website](https://changgao.site/)
* **Manish Prajapat**  *Department of Mechanical and Process Engineering, ETH Zurich*
* **Frederic Debraine** - *Department of Mechanical and Process Engineering, ETH Zurich*
* **Martyna Dziadosz** - *Department for BioMedical Research, University of Bern*

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
