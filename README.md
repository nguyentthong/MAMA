# MAMA: A Meta-optimized Angular Margin Contrastive Framework for Video-Language Representation Learning from Large Vision Language Model


[**MAMA: A Meta-optimized Angular Margin Contrastive Framework for Video-Language Representation Learning from Large Vision Language Model**](https://arxiv.org/abs/2407.03788)
Thong Nguyen, Yi Bin, Xiaobao Wu, Xinshuai Dong, Zhiyuan Hu, Khoi Le, Cong-Duy Nguyen, See-Kiong Ng, Luu Anh Tuan
ECCV 2024

[arxiv](https://arxiv.org/abs/2407.03788) | [bibtex](#citing-mama) | [🤗 demo](https://huggingface.co/spaces/thongnguyen5999/mama) | [website](https://nguyentthong.github.io/mama)

MAMA (**M**eta-optimized **A**ngular **MA**rgin Contrastive Framework for Video-Language Representation Learning from Large Vision Language Model) is a novel approach to learn video-language representations from Large Vision-Language Model (LVLM). We utilize LLaVA to augment training video-text data, and utilize an angular margin-based contrastive learning combined with meta-learning to optimize the effectiveness of the LLaVA-augmented data.

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

## Requirements
- transformers==4.31.0 
- tqdm==4.66.2
- torchvision==0.15.2
- torch==2.0.1
- toolz==0.12.1
- scikit-learn==1.2.2
- scikit-image==0.22.0
- sentencepiece==0.1.99
- easydict==1.12
- ete3==3.1.3
- fairscale==0.4.13
- einops==0.6.1
- opencv-python==4.9.0.80
- peakutils==1.3.4
- pandas==2.2.1
- scipy==1.12.0
- apex=0.1
- deepspeed==0.12.4
- deprecated==1.2.14

## How to run
### 1) Video question answering
- Download the folders and put into the directory `videoQA`.
- Execute the training and finally evaluation via `bash ./scripts/run_videoQA.sh`

### 2) Text-video retrieval
- Download the folders and zipped file, then put them into the directory `text_video_retrieval` (remember to unzip the file).
- Execute the training and finally evaluation via `bash ./scripts/run_text_video_retrieval.sh`

### 3) Run the demo of LVLM
- Put the video into the directory `assets`
- Execute the generation code `bash ./scripts/demo_lvlm.sh`
