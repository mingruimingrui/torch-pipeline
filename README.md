
**torch-pipeline** is a collection of training scripts making use of `torch-collections` and `torch-datasets`.

This repository contains optimized training scripts for training models from torch-collections on popular deep learning datasets leveraging off the torch-datasets high level APIs.

## Installation of requirements

Scripts in this repository assumes that these packages are installed,

* torch >= (latest stable)
* torchvision >= (latest stable)
* cv2 >= 3.0.0
* pillow >= 3.0.0

It is important to follow the latest stable of [`pytorch`](https://pytorch.org/) as it is still in prerelease and every patch adds important bug fixes.

Earlier versions of `torch` and `torchvision` would probably not work but earlier versions of other packages probably can work, however have not been tested before.

Additionally the `torch-collections` and `torch-datasets` has to be installed. The installation instructions to install the latest-stable versions of these 2 package can be found in `install_requirements.sh`. Simply run `./install_requirements.sh`.
