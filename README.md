
<div align="center">

<p align="center">
  <img src="assets/logo.png" alt="SVI" width="400"/>
</p>

<h1>Stable Video Infinity: Infinite-Length Video Generation with Error Recycling (Wan 2.2 14B)</h1>

[Wuyang Li](https://wymancv.github.io/wuyang.github.io/) · [Wentao Pan](https://scholar.google.com/citations?user=sHKkAToAAAAJ&hl=zh-CN) · [Po-Chien Luan](https://scholar.google.com/citations?user=Y2Oth4MAAAAJ&hl=zh-TW) · [Yang Gao](https://scholar.google.com/citations?user=rpT0Q6AAAAAJ&hl=en) · [Alexandre Alahi](https://scholar.google.com/citations?user=UIhXQ64AAAAJ&hl=en)

[VITA@EPFL](https://www.epfl.ch/labs/vita/)

<a href='https://stable-video-infinity.github.io/homepage/'><img src='https://img.shields.io/badge/Project-Page-green'></a>
<a href='https://arxiv.org/abs/2510.09212'><img src='https://img.shields.io/badge/Technique-Report-red'></a>
<a href='https://huggingface.co/vita-video-gen/svi-model/tree/main/version-1.0'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a>
<a href='https://huggingface.co/datasets/vita-video-gen/svi-benchmark'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-orange'></a>

Technical introduction (unofficial): [AI Papers Slop (English)](https://www.youtube.com/watch?v=vKPCqPsCfZg); [WechatApp (Chinese)](https://mp.weixin.qq.com/s?__biz=MzIwMTE1NjQxMQ==&mid=2247641601&idx=1&sn=e86ae40b54fda22eda2ebd818b38de73&chksm=978a0c69a14a79192b1ca81f257f093362add316acdcdff69c67ab5d186f8af7f8e84931632a&mpshare=1&srcid=1016e1aTWfR71TRJJHDFgMHf&sharer_shareinfo=273ee623f20eba9542ff4b8c3a0c35d1&sharer_shareinfo_first=559e5442227d44f61573005b4e12d83c&from=timeline&scene=2&subscene=2&clicktime=1761249340&enterid=1761249340&sessionid=0&ascene=45&fasttmpl_type=0&fasttmpl_fullversion=7965100-zh_CN-zip&fasttmpl_flag=0&realreporttime=1761249340647#rd)
</div>



We really appreciate the beautiful videos created by [@RuneGjerde](https://github.com/RuneGjerde) using SVI!

<table>
  <tr>
    <td width="50%">
      <video src="https://github.com/user-attachments/assets/568edfc7-5aaf-4f9a-ba21-9226e161a227"
             controls
             muted
             width="100%">
      </video>
    </td>
    <td width="50%">
      <video src="https://github.com/user-attachments/assets/a5a23659-90fb-475f-bfff-6b08ea67d294"
             controls
             muted
             width="100%">
      </video>
    </td>
  </tr>
</table



## ✨ New Features

1. **Better dynamics:** Compared with the Wan 2.1 version, SVI 2.0 Pro produces more dynamic and natural motion, thanks to the inherent capabilities of Wan 2.2.

2. **Cross-clip consistency:** This version provides a certain level of cross-clip consistency. As shown in the demo, your cat is still your cat: Even when a character completely leaves the frame in one clip and reappears several clips later, the model maintains a reasonable degree of visual consistency.

SVI 2.0 Gallery
<table>
  <tr>
    <td>
      <video src="https://github.com/user-attachments/assets/7bdb3120-ec18-4def-9356-49ebf95293f3"
             controls
             muted
             width="100%">
      </video>
      <p align="center">Your cat is still your cat</p>
    </td>
    <td>
      <video src="https://github.com/user-attachments/assets/fd88fc44-38e9-4972-ad41-5f384eec8191"
             controls
             muted
             width="100%">
      </video>
      <p align="center">Your dog can run anywhere</p>
    </td>
    <td>
      <video src="https://github.com/user-attachments/assets/6172b2a0-7f77-490e-9fd2-2f372aba936a"
             controls
             muted
             width="100%">
      </video>
      <p align="center">Your baby is still your baby</p>
    </td>
  </tr>
</table>

SVI 2.0 Pro Gallery
- More dynamic and expressive motions  
- Support for a wider range of scene transitions
- More interesting

<table>
  <tr>
    <td>
      <video src="https://github.com/user-attachments/assets/e6d270ab-6992-4b60-a12a-6512b2e5dd33"
             controls
             muted
             width="100%">
      </video>
    </td>
    <td>
      <video src="https://github.com/user-attachments/assets/6070774e-f4fd-4194-ad7c-731859239395"
             controls
             muted
             width="100%">
      </video>
    </td>
    <td>
      <video src="https://github.com/user-attachments/assets/4709eb7c-ac4c-4c12-a4f0-20a91d2a64ad"
             controls
             muted
             width="100%">
      </video>
    </td>
  </tr>
</table>



<table>
  <tr>
    <td>
      <video src="https://github.com/user-attachments/assets/cfc630aa-ee03-4909-ad81-1dac85b63c99"
             controls
             muted
             width="100%">
      </video>
    </td>
    <td>
      <video src="https://github.com/user-attachments/assets/231cff4e-fd22-415e-ab50-fc996e3160e7"
             controls
             muted
             width="100%">
      </video>
    </td>
    <td>
      <video src="https://github.com/user-attachments/assets/1c829fe1-550c-4075-8a6c-419b232c1c85"
             controls
             muted
             width="100%">
      </video>
    </td>
  </tr>
</table>





## 🚀 News about Wan 2.2-based SVI

- **[26 Dec 2025] SVI 2.0 Pro Released:** More details can be found [svi_2.0_pro.md](docs/svi/svi_2.0_pro.md).

- **[4 Dec 2025] SVI 2.0 Released**  
 

<table>
  <tr>
    <td width="50%">
      <video src="https://github.com/user-attachments/assets/7bdb3120-ec18-4def-9356-49ebf95293f3"
             controls
             muted
             width="100%">
      </video>
      <p align="center">SVI 2.0</p>
    </td>
    <td width="50%">
      <video src="https://github.com/user-attachments/assets/231cff4e-fd22-415e-ab50-fc996e3160e7"
             controls
             muted
             width="100%">
      </video>
      <p align="center">SVI 2.0 Pro</p>
    </td>
  </tr>
</table>

## 😀 ComfyUI Users

- **[26 Dec 2025] SVI 2.0 Pro**: We have redesigned some core components of SVI 2.0 Pro, so this version is no longer compatible with the original workflow. More details can be found [svi_2.0_pro.md](docs/svi/svi_2.0_pro.md).

- **[10 Dec 2025] SVI 2.0**: check out our preview workflow: `Stable-Video-Infinity/comfyui_workflow`! Unfortunately, we noticed a conflict between the LightX2V LoRA and the SVI LoRA: more details can be found [here](./docs/svi/comfyui.md).


## ❓More Information

1. **Platform:** This branch is built on the updated Diffsynth 2.0, so the environment needs to be reconfigured accordingly. P.S. Great thanks to the Diffsynth team for their outstanding codebase maintenance.

2. **Re-implementation Tips:** To enhance dynamics, particularly exit–reenter consistency, we introduce a simple yet effective modification: following the SVI-Shot training setup, we ensure that the randomly sampled padding frame never appears in the currently generated video clips. For example, we may use frames 1–81 for generation and reserve frame 100 exclusively for padding. In addition, we also apply strong image augmentation to the first frame to encourage the model to perform restoration guided by the padding (i.e., the anchor).

3. **Tips for Generating Better Long Videos:** Please refer to [tips.md](./docs/svi/tips.md).


</div>

**📧 Contact**: [wuyang.li@epfl.ch](mailto:wuyang.li@epfl.ch)


## 🔧 Environment Setup

The original docs of diffsynth 2.0 is [here](docs/README.md). Using different PyTorch versions leads to different results even when using the same random seed. Our current environment uses torch==2.7.1.


```bash
git clone https://github.com/vita-epfl/Stable-Video-Infinity.git -b svi_wan22

conda create -n svi_wan22 python=3.10 
conda activate svi_wan22

pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128

cd Stable-Video-Infinity

pip install -e .

pip install flash_attn==2.8.0.post2
```

## 📦 Model Preparation

| Model                           | Task                    | Input                      | Output           | Hugging Face Link                                                                                                                | Comments                                                                                                   |
| ------------------------------- | ----------------------- | -------------------------- | ---------------- | -------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| **SVI (Wan 2.2 14B)**              | Single-scene (suppors some transitions) | Image + Text prompt stream        | Long video       | [🤗 High  Noise Model](https://huggingface.co/vita-video-gen/svi-model/resolve/main/version-2.0/SVI_Wan2.2-I2V-A14B_high_noise_lora_v2.0.safetensors) <br> [🤗 Low  Noise Model](https://huggingface.co/vita-video-gen/svi-model/resolve/main/version-2.0/SVI_Wan2.2-I2V-A14B_low_noise_lora_v2.0.safetensors)            | Generate consistent long video with 1 text prompt stream.                            |                                      |                                  |
| **SVI Pro (Wan 2.2 14B)**             | Single-scene (suppors some transitions) | Image + Text prompt stream        | Long video       | [🤗 High  Noise Model](https://huggingface.co/vita-video-gen/svi-model/resolve/main/version-2.0/SVI_Wan2.2-I2V-A14B_high_noise_lora_v2.0_pro.safetensors) <br> [🤗 Low  Noise Model](https://huggingface.co/vita-video-gen/svi-model/resolve/main/version-2.0/SVI_Wan2.2-I2V-A14B_low_noise_lora_v2.0_pro.safetensors)            | Generate consistent long video with 1 text prompt stream.                            |                                      |                                  |


```bash
# login with your fine-grained token
huggingface-cli login

# SVI 2.0 Pro (Wan2.2 14B)
huggingface-cli download vita-video-gen/svi-model --local-dir ./models/Stable-Video-Infinity --include "version-2.0/SVI_Wan2.2-I2V-A14B_high_noise_lora_v2.0_pro.safetensors"

huggingface-cli download vita-video-gen/svi-model --local-dir ./models/Stable-Video-Infinity --include "version-2.0/SVI_Wan2.2-I2V-A14B_low_noise_lora_v2.0_pro.safetensors"

# SVI 2.0 (Wan2.2 14B)
huggingface-cli download vita-video-gen/svi-model --local-dir ./models/Stable-Video-Infinity --include "version-2.0/SVI_Wan2.2-I2V-A14B_high_noise_lora_v2.0.safetensors"

huggingface-cli download vita-video-gen/svi-model --local-dir ./models/Stable-Video-Infinity --include "version-2.0/SVI_Wan2.2-I2V-A14B_low_noise_lora_v2.0.safetensors"
```

## 🎮 Play with Wan 2.2-SVI

SVI 2.0 Pro: 50-clips 250-second generation.

```bash
CUDA_VISIBLE_DEVICES=0 python inference_svi_2.0_pro.py \
    --output_root videos \
    --height 832 \
    --width  480 \
    --lora_path_high models/Stable-Video-Infinity/version-2.0/SVI_Wan2.2-I2V-A14B_high_noise_lora_v2.0_pro.safetensors \
    --lora_path_low models/Stable-Video-Infinity/version-2.0/SVI_Wan2.2-I2V-A14B_low_noise_lora_v2.0_pro.safetensors \
    --fps 15 \
    --ref_image_path ./data/toy_test/demo1/frame.png \
    --prompt_path ./data/toy_test/demo1/prompt.txt \
    --num_clips 50 \
    --cfg_scale 4.0 \
    --num_overlap_frame 5 \
    --num_motion_latent 1
```

SVI 2.0: By using the following command, SVI should be able to generate the [demo video](assets/demo_480p.mp4).

```bash
# This is consistent with SVI-Shot
CUDA_VISIBLE_DEVICES=0 python inference_svi_2.0.py \
    --output_root videos \
    --height 480 \
    --width 832 \
    --lora_path_high models/Stable-Video-Infinity/version-2.0/SVI_Wan2.2-I2V-A14B_high_noise_lora_v2.0.safetensors \
    --lora_path_low models/Stable-Video-Infinity/version-2.0/SVI_Wan2.2-I2V-A14B_low_noise_lora_v2.0.safetensors \
    --fps 15 \
    --ref_image_path ./data/toy_test/demo2/frame.jpg \
    --prompt_path ./data/toy_test/demo2/prompt.txt \
    --num_clips 10 
```

If you experience the slow download speed from Modelscope, you can manually download the models from Huggingface and organize them as follows:

```bash
models/
 ├── DiffSynth-Studio/
 │   └── Wan-Series-Converted-Safetensors/
 │       ├── models_t5_umt5-xxl-enc-bf16.safetensors
 │       └── Wan2.1_VAE.safetensors
 ├── Stable-Video-Infinity/
 │   └── version-2.0/
 │       ├── SVI_Wan2.2-I2V-A14B_high_noise_lora_v2.0.safetensors
 │       ├── SVI_Wan2.2-I2V-A14B_low_noise_lora_v2.0.safetensors
 │       ├── SVI_Wan2.2-I2V-A14B_high_noise_lora_v2.0_pro.safetensors
 │       └── SVI_Wan2.2-I2V-A14B_low_noise_lora_v2.0_pro.safetensors
 └── Wan-AI/
     ├── Wan2.1-T2V-1.3B/
     │   └── google/
     │       └── umt5-xxl/
     │           ├── special_tokens_map.json
     │           ├── spiece.model
     │           ├── tokenizer_config.json
     │           └── tokenizer.json
     └── Wan2.2-I2V-A14B/
         ├── high_noise_model/
         │   ├── diffusion_pytorch_model-00001-of-00006.safetensors
         │   ├── diffusion_pytorch_model-00002-of-00006.safetensors
         │   ├── diffusion_pytorch_model-00003-of-00006.safetensors
         │   ├── diffusion_pytorch_model-00004-of-00006.safetensors
         │   ├── diffusion_pytorch_model-00005-of-00006.safetensors
         │   └── diffusion_pytorch_model-00006-of-00006.safetensors
         └── low_noise_model/
             ├── diffusion_pytorch_model-00001-of-00006.safetensors
             ├── diffusion_pytorch_model-00002-of-00006.safetensors
             ├── diffusion_pytorch_model-00003-of-00006.safetensors
             ├── diffusion_pytorch_model-00004-of-00006.safetensors
             ├── diffusion_pytorch_model-00005-of-00006.safetensors
             └── diffusion_pytorch_model-00006-of-00006.safetensors
```

## ❤️ Citation

If you find our work helpful for your research, please consider citing our paper. Thank you so much!

```bibtex
@article{li2025stable,
  title={Stable Video Infinity: Infinite-Length Video Generation with Error Recycling},
  author={Li, Wuyang and Pan, Wentao and Luan, Po-Chien and Gao, Yang and Alahi, Alexandre},
  journal={arXiv preprint arXiv:2510.09212},
  year={2025}
}
```
