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
