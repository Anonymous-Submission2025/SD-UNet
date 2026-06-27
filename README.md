<h2 align="center">✨SD-UNet: Integrating Structure Tensor and Constrained Cascaded Focusing for Medical Image Segmentation</h2>

<p align="center">
  <b>[Anonymous Submission for Double-Blind Review]</b>
</p>

---

**🚨 Note to Reviewers (Reproducibility & Open Source Plan):**
To strictly comply with the double-blind review policy and protect core intellectual property prior to publication, this repository currently provides an **Inference Demo** with pre-trained weights (traced/compiled). This allows for quick verification of our model's performance on the sample data. 
**The complete training framework, core model definition scripts, and data pre-processing pipelines will be fully open-sourced immediately upon acceptance.**

---

## Overview 🔍
<div>
    <img src="asserts/model2.pdf" width="96%" height="96%">
</div>

**Figure 1. The framework of the proposed SD-UNet.**

**_Abstract -_** In medical image segmentation, lesions are often intertwined with complex anatomical structures, leading to significant semantic co-occurrence interference. Existing methods often perform decoupling and filtering in the deep feature space, but this is not the optimal solution. This is because low-level features have already been contaminated by background information during the initial extraction phase, leading to early feature aliasing. Moreover, conventional directional convolutions rely on predefined discrete rotation angles, lacking geometric adaptability to irregular lesions. To overcome the aforementioned limitations, this paper proposes the SD-UNet architecture. Firstly, this paper proposes a Dynamic Cascaded Refinement Module (DCRM) based on a constrained cascaded focusing strategy. This module uses the initial mask as a prior, dynamically modulating the local feature aggregation weight regions and guiding edge refinement; this not only helps the model accurately focus on the core lesions but also blocks the transmission of noise at its source. Secondly, this paper designs the Structure Tensor Convolution (STConv). STConv constructs grouped structure tensors by calculating local spatial derivatives and solves for the principal eigenvector to derive the adaptive optimal direction; this mechanism achieves precise characterization of irregular lesion contours through dynamically rotating convolution kernel groups. Comprehensive evaluations on five public benchmark datasets indicate that SD-UNet has achieved SOTA performance, with its average IoU and average Dice metrics surpassing the strongest baseline model by 2.87\% and 1.70\%, respectively. These results demonstrate robust representation and generalization capability.  



## Getting Started 🚀

### 1. Install Environment
We provide a streamlined environment setup for the inference demo. 

```bash
conda create -n SD-UNet python=3.10
conda activate SD-UNet
# Install PyTorch (adjust the CUDA version according to your hardware)
pip3 install torch torchvision --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
# Install dependencies for inference and visualization
pip install opencv-python numpy scikit-image matplotlib
