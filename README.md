# SCPC 2025 AI 

- 2025 Samsung Collegiate Programming Challenge : AI 
- Submission Code 
- Huigyoeng Son 

### environment 

System 

- OS: Linux
- GPU: 1x NVIDIA A100 SXM4 (40GB VRAM) 
- CUDA: 12.8

Python Environment 

- Python: 3.12.11
- All dependencies are managed via pip.

Key Libraries 

| Package                       | Version      |
| ----------------------------- | ------------ |
| torch                         | 2.7.1+cu128  |
| torchaudio                    | 2.7.1+cu128  |
| torchvision                   | 0.22.1+cu128 |
| transformers                  | 4.54.0       |
| accelerate                    | 1.9.0        |
| peft                          | 0.16.0       |
| bitsandbytes                  | 0.46.1       |
| trl                           | 0.19.1       |
| datasets                      | 4.0.0        |
| transformers-stream-generator | 0.0.5        |
| safetensors                   | 0.5.3        |
| einops                        | 0.8.1        |
| tiktoken                      | 0.9.0        |
| diffusers                     | 0.20.0       |
| numpy                         | 2.1.2        |
| pandas                        | 2.3.1        |
| pillow                        | 11.0.0       |
| tqdm                          | 4.67.1       |


Jupyter (Optional)

| Package         | Version |
| --------------- | ------- |
| ipykernel       | 6.30.0  |
| jupyter\_client | 8.6.3   |
| jupyter\_core   | 5.8.1   |

Installation

- Install dependencies with requirements.txt

  ```
  pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
  ```

- Install dependencies individually

  ```

  # Install PyTorch (CUDA 12.1)
  pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

  # Core libraries
  pip install transformers==4.54.0 accelerate==1.9.0 peft==0.16.0 bitsandbytes==0.46.1
  pip install trl==0.19.1 datasets==4.0.0 tqdm==4.67.1
  pip install transformers-stream-generator==0.0.5 einops==0.8.1 tiktoken==0.9.0
  pip install diffusers==0.20.0 safetensors==0.5.3

  # Utilities
  pip install numpy==2.1.2 pandas==2.3.1 pillow==11.0.0

  # Jupyter (optional)
  pip install ipykernel==6.30.0 jupyter_client==8.6.3 jupyter_core==5.8.1

  ```

### Run

Generate dataset

1. run generate_scene_prompt.py
2. run generate_scene_image.py
3. run generate_question_answer.py

Train & Inference

1. run train.ipynb
2. run inference.ipynb
