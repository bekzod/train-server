# train-server

#### Env variables needed
create .env file with following
HF_HUB_ENABLE_HF_TRANSFER=1
CUDA_VISIBLE_DEVICES=0,1
HF_HUB_VERBOSITY=info
HF_TOKEN=token

## Installation

### System dependencies
sudo apt update && sudo apt install -y ffmpeg libsndfile1 git
### Install NVIDIA CUDA Toolkit (if not already installed)
sudo apt install -y nvidia-cuda-toolkit

### Python dependencies
To install and set up the project, follow these steps:
1. Create and activate a virtual environment:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
