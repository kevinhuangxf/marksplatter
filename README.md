# MarkSplatter

The official implementation for [MarkSplatter: Generalizable Watermarking for 3D Gaussian Splatting Model via Splatter Image Structure](https://arxiv.org/abs/2509.00757) (ACM MM 2025)

## Installation

```shell
conda create -n marksplatter python=3.10

pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# a modified gaussian splatting (+ depth, alpha rendering)
git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
pip install ./diff-gaussian-rasterization

pip install -r requirements.txt
```

## Download

Please use the download script to download the MarkSplatter and GaussianBridge checkpoint.

```shell
python download.py
```

MarkSplatter can also support using the [LGM](https://github.com/3DTopia/LGM/tree/main) official checkpoint. Please refer to their homepage to download their pretrained model model_fp16_fixrot.safetensors and place under the ckpts folder.

## Inference Examples

Run the example with GaussianBridge:

```shell
python infer.py big \
--resume ./ckpts/gaussianbridge.safetensors \
--marksplatter_ckpt_path ./ckpts/marksplatter.safetensors \
--workspace ./workspace_test \
--test_path examples/rabbit
```

Run the example with LGM:

```shell
python infer.py big \
--resume ./ckpts/model_fp16_fixrot.safetensors \
--marksplatter_ckpt_path ./ckpts/marksplatter.safetensors \
--workspace ./workspace_test \
--test_path examples/bird
```

## Acknowledgement

This project is based on the [LGM](https://github.com/3DTopia/LGM/tree/main) and [Watermark-anything](https://github.com/facebookresearch/watermark-anything) repositories. If you are insterested in our work, please also consider to cite them.
