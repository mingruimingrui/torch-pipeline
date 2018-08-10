
**torch-pipeline** is a collection of training scripts making use of `torch-collections` and `torch-datasets`.

This repository contains optimized training scripts for training models from torch-collections on popular deep learning datasets leveraging off the torch-datasets high level APIs.

## Installation of requirements

Scripts in this repository assumes that these packages are installed,

* torch >= 0.4.0
* torchvision >= 0.2.1
* cv2 >= 3.0.0
* pillow >= 3.0.0

Earlier versions can work, however have not been tested before.

Additionally the `torch-collections` and `torch-datasets` has to be installed. The installation instructions to install the latest-stable versions of these 2 package can be found in `install_requirements.sh`. Simply run `./install_requirements.sh`.
