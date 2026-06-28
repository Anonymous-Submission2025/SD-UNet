<h2 align="center">✨SD-UNet: Integrating Structure Tensor and Constrained Cascaded Focusing for Medical Image Segmentation</h2>

<p align="center">
  <b>[Anonymous Submission for Double-Blind Review]</b>
</p>

---

**🚨 Note to Reviewers (Reproducibility & Open Source Plan):**
To strictly comply with the double-blind review policy and protect core intellectual property prior to publication, this repository provides a fully functional **Inference Demo**. 
* **Encapsulation**: Core model logic is compiled as Python bytecode to ensure binary-level source protection while maintaining full execution compatibility.
* **Reproducibility**: We have provided pre-trained weights and sample data. Running the demo will automatically verify the model's performance (IoU/Dice) on the provided dataset.
* **Commitment**: The complete training framework, full-scale training scripts, and comprehensive data processing pipelines will be 100% open-sourced immediately upon acceptance.

## Overview 🔍

<p align="center">
  <img src="asserts\model.png" width="80%">
</p>

**Figure 1. The framework of the proposed SD-UNet.**


**_Abstract -_** In medical image segmentation, lesions are often intertwined with complex anatomical structures, leading to significant semantic co-occurrence interference. Existing methods often perform decoupling and filtering in the deep feature space, but this is not the optimal solution. This is because low-level features have already been contaminated by background information during the initial extraction phase, leading to early feature aliasing. Moreover, conventional directional convolutions rely on predefined discrete rotation angles, lacking geometric adaptability to irregular lesions. To overcome the aforementioned limitations, this paper proposes the SD-UNet architecture. Firstly, this paper proposes a Dynamic Cascaded Refinement Module (DCRM) based on a constrained cascaded focusing strategy. This module uses the initial mask as a prior, dynamically modulating the local feature aggregation weight regions and guiding edge refinement; this not only helps the model accurately focus on the core lesions but also blocks the transmission of noise at its source. Secondly, this paper designs the Structure Tensor Convolution (STConv). STConv constructs grouped structure tensors by calculating local spatial derivatives and solves for the principal eigenvector to derive the adaptive optimal direction; this mechanism achieves precise characterization of irregular lesion contours through dynamically rotating convolution kernel groups. Comprehensive evaluations on five public benchmark datasets indicate that SD-UNet has achieved SOTA performance, with its average IoU and average Dice metrics surpassing the strongest baseline model by 2.87\% and 1.70\%, respectively. These results demonstrate robust representation and generalization capability.  

## Getting Started 🚀

### 1. Environment Setup
We provide a streamlined environment setup for the inference demo.

```bash
conda create -n SD-UNet python=3.10 -y
conda activate SD-UNet
# Install core dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install timm triton opencv-python numpy pillow tifffile tqdm
