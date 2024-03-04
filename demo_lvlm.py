import shutil
from video_keyframe_detector.cli import keyframeDetection
import argparse
import os
import numpy as np
import cv2
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token, KeywordsStoppingCriteria, process_images
from llava.model.builder import load_pretrained_model
from transformers import BertTokenizerFast
from llava.utils import disable_torch_init
from llava.constants import DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from PIL import Image
import torch


def image_parser(args):
    out = args.image_file.split(args.sep)
    return out

def load_image(image_file):
    image = Image.open(image_file).convert("RGB")
    return image


def extract_keyframes(video_path):
    if os.path.exists("temp"): shutil.rmtree("temp")
    temp_folder = "temp"
    target_folder = "video_frames"
    os.makedirs("temp", exist_ok=True)
    keyframeDetection(video_path, temp_folder, 0.6)
    num_frames = 12

    video_frame_list = sorted(os.listdir("temp"), key=lambda x: int(x.split('.')[0].split('_')[-1]))
    os.makedirs(target_folder, exist_ok=True)
    selected_frame_idx_set = set(np.linspace(1, len(video_frame_list)-1, num_frames).astype(int))
    cnt = 0
    for i in range(len(video_frame_list)):
        if i in selected_frame_idx_set:
            source_file = os.path.join(temp_folder, video_frame_list[i])
            target_file = os.path.join(target_folder, "frame_{}.jpg".format(cnt))
            shutil.copyfile(source_file, target_file)
            cnt += 1

    shutil.rmtree("temp", ignore_errors=True)


def concatenate_keyframes():
    keyframe_filename_list = os.listdir("video_frames")
    keyframe_list = []
    for keyframe_filename in keyframe_filename_list:
        keyframe = cv2.imread(os.path.join("video_frames", keyframe_filename))
        keyframe_list.append(keyframe)

    keyframe_row1 = cv2.hconcat(keyframe_list[:4])
    keyframe_row2 = cv2.hconcat(keyframe_list[4:8])
    keyframe_row3 = cv2.hconcat(keyframe_list[8:])
    img = cv2.vconcat([keyframe_row1, keyframe_row2, keyframe_row3])
    cv2.imwrite("concatenated_keyframe_image.jpg", img)

def eval_model(args, model_name, tokenizer, model, image_processor, context_len):
    # Model
    DEFAULT_IMAGE_TOKEN = "<image>"
    disable_torch_init()

    qs = args.query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN

    if model.config.mm_use_im_start_end:
        qs = image_token_se + "\n" + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image_files = image_parser(args)
    image = load_image(image_files[0])
    images_tensor = process_images([image], image_processor, model.config).to(model.device, dtype=torch.float16)

    input_ids = (tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda())

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(input_ids, images=images_tensor, do_sample=True, temperature=0.2, max_new_tokens=1024, use_cache=True, stopping_criteria=[stopping_criteria],)

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids")
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()
    return outputs

def generate_description():
    model_path = "liuhaotian/llava-v1.5-7b"
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)
    bert_tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    prompt = "Write a caption sentence for the video in order from left to right, top to bottom."
    image_file = "concatenated_keyframe_image.jpg"

    args = type('Args', (), {
        "model_path": model_path,
        "model_base": None,
        "model_name": model_name,
        "query": prompt,
        "conv_mode": None,
        "image_file": image_file,
        "sep": ",",
        "max_new_tokens": 1024,
        "max_seq_len": 40,
        "temperature": 0.2
    })()

    image_caption_len = 10e6
    num_repeated_times = 10
    while image_caption_len > 40:
        image_caption = eval_model(args, model_name, tokenizer, model, image_processor, context_len)
        image_caption_tokens = bert_tokenizer(image_caption)
        image_caption_len = len(image_caption_tokens['input_ids'])
        num_repeated_times += 1
        if num_repeated_times == 10: break
    
    print(image_caption)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-path", type=str)
    args = parser.parse_args()

    extract_keyframes(args.video_path)
    concatenate_keyframes()
    generate_description()

if __name__ == '__main__':
    main()