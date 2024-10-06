# MAMA: A Meta-optimized Angular Margin Contrastive Framework for Video-Language Representation Learning from Large Vision Language Model


[**MAMA: A Meta-optimized Angular Margin Contrastive Framework for Video-Language Representation Learning from Large Vision Language Model**](https://arxiv.org/abs/2407.03788)

Thong Nguyen, Yi Bin, Xiaobao Wu, Xinshuai Dong, Zhiyuan Hu, Khoi Le, Cong-Duy Nguyen, See-Kiong Ng, Luu Anh Tuan
ECCV 2024

[arxiv](https://arxiv.org/abs/2407.03788) | [bibtex](#citing-mama) | [🤗 demo](https://huggingface.co/spaces/thongnguyen5999/mama) | [website](https://nguyentthong.github.io/mama)

**MAMA** (**M**eta-optimized **A**ngular **MA**rgin Contrastive Framework for Video-Language Representation Learning from Large Vision Language Model) is a novel approach to learn video-language representations from Large Vision-Language Model (LVLM). We utilize LLaVA to augment training video-text data, and utilize an angular margin-based contrastive learning combined with meta-learning to optimize the effectiveness of the LLaVA-augmented data.

**Sample Generation:**

| Video | Generation |
| --------|-------------|
| <img src="assets/mixkit-pastry-chef-cutting-a-loaf-into-slices-43015-medium.gif" height=128> | so now we're going to slice the bread |
| <img src="assets/mixkit-pastry-chef-cutting-a-loaf-into-slices-43015-medium.gif" height=128> | so now we're going to slice the bread |

[Try out](#lvlm-demo) our LVLM-based pipeline to generate text descriptions for your own videos! 
You can also try out a web demo here: [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/thongnguyen5999/mama)

The resulting video-language model sets a new state-of-the-art on a number of popular video tasks!

## Introduction and installation

<span style="font-variant:small-caps;">MAMA</span> leverages Large Vision-Language Models (LVLM) as to automatically augment video-text training data, and uses these data to fine-tune strong video-language models.

<img src="assets/lavila_ego4d.gif" height=384> 

See [INSTALL.md](docs/INSTALL.md) to install this code.

## MAMA

MAMA consists of a subtractive angular margin contrastive objective, powered by meta-learning to weigh the important of the training video-text data.

<img src="assets/mama_illustration.png" height=384>

## Citing MAMA

```bibtex
@article{nguyen2024meta,
  title={Meta-optimized Angular Margin Contrastive Framework for Video-Language Representation Learning},
  author={Nguyen, Thong and Bin, Yi and Wu, Xiaobao and Dong, Xinshuai and Hu, Zhiyuan and Le, Khoi and Nguyen, Cong-Duy and Ng, See-Kiong and Tuan, Luu Anh},
  journal={arXiv preprint arXiv:2407.03788},
  year={2024}
}
```
