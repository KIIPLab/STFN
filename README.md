# STFN : Swin Transformer Fusion Network for Image Quality Assessment
https://github.com/KIIPLab/STFN



> **Abstract:** *This paper presents an efficient deep-learning model named Swin Transformer fusion network
(STFN) for full-reference image quality assessment (FR-IQA). The STFN model uses the first and second
stages of the Swin Transformer for feature extraction. To unify the features from these two stages, we
propose fusion operations including reverse patch merging (RPM) and mediator block (MB) operations.
The RPM is a kind of reverse operation of the patch merging operation in the Swin Transformer stage, and
it reshapes the size of the second stage feature so as to match to that of the first stage feature. The MB
operation efficiently combines multiple features from the RPM block and the first stage Swin Transformer
for subsequent operations. Experimental results show that the proposed STFN model provides significantly
improved performance than the previous traditional and deep-learning models for various kinds of image
datasets for FR-IQA. The STFN model also shows superior performance compared to the state-of-the-art
method for FR-IQA with smaller training time and model size.* 



## Getting Started

### Prerequisites
- windows 10
- NVIDIA GPU + CUDA CuDNN
- Python 3.7




## Acknowledgment
The codes borrow heavily from AHIQ implemented by [IIGROUP] https://github.com/IIGROUP/AHIQ and I really appreciate it.

