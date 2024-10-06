import argparse
import shutil
import os
from video_keyframe_detector.cli import keyframeDetection
import numpy as np
import cv2
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from PIL import Image
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)
import torch


def extract_keyframes(video_path, num_keyframes=12):
    video_id = video_path.split('/')[-1].strip().split('.')[0]

    os.makedirs("temp", exist_ok=True)

    keyframeDetection(video_path, "temp", 0.6)
    video_frame_list = sorted(os.listdir(os.path.join("temp", "keyFrames")), key=lambda x: int(x.split('.')[0][8:]))
    os.makedirs(os.path.join("video_frames", video_id), exist_ok=True)
    selected_frame_idx_set = set(np.linspace(1, len(video_frame_list) - 1, num_keyframes).astype(int))
    cnt = 0
    for i in range(len(video_frame_list)):
        if i in selected_frame_idx_set:
            source_file = os.path.join("temp", "keyFrames", video_frame_list[i])
            target_file = os.path.join("video_frames", video_id, f"frame_{cnt}.jpg")
            shutil.copyfile(source_file, target_file)
            cnt += 1
    
    shutil.rmtree("temp", ignore_errors=True)


def concatenate_frames(video_path):
    os.makedirs("concatenated_frames", exist_ok=True)
    video_id = video_path.split('/')[-1].strip().split('.')[0]
    image_frame_dir = os.path.join("video_frames", video_id)
    image_frame_list = sorted(os.listdir(os.path.join(image_frame_dir)), key=lambda x: int(x.split('.')[0].split('_')[1]))
    img_list = []
    for image_frame in image_frame_list:
        img_frame = cv2.imread(os.path.join(image_frame_dir, image_frame))
        img_list.append(img_frame)

    img_row1 = cv2.hconcat(img_list[:4])
    img_row2 = cv2.hconcat(img_list[4:8])
    img_row3 = cv2.hconcat(img_list[8:12])

    img_v = cv2.vconcat([img_row1, img_row2, img_row3])
    cv2.imwrite(os.path.join("concatenated_frames", f"{video_id}.jpg"), img_v)


def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

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
    images = load_images(image_files)
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(
            f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
        )
    outputs = tokenizer.batch_decode(
        output_ids[:, input_token_len:], skip_special_tokens=True
    )[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()
    return outputs


def generate_video_caption(video_path):
    model_path = "liuhaotian/llava-v1.5-7b"
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)
    video_id = video_path.split('/')[-1].strip().split('.')[0]

    image_file = os.path.join("concatenated_frames", f"{video_id}.jpg")
    prompt = "In a short paragraph, describe the process in the video."

    args = type('Args', (), {
        "model_path": model_path,
        "model_base": None,
        "model_name": get_model_name_from_path(model_path),
        "query": prompt,
        "conv_mode": None,
        "image_file": image_file,
        "sep": ",",
        "max_new_tokens": 1024,
        "temperature": 0.2
    })()

    video_caption = eval_model(args, model_name, tokenizer, model, image_processor, context_len).replace("images", "video").replace("image", "video")
    print(f"The video caption: {video_caption}")

def clean_files_and_folders():
    shutil.rmtree("concatenated_frames")
    shutil.rmtree("video_frames")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-path", type=str)
    args = parser.parse_args()

    extract_keyframes(args.video_path)
    concatenate_frames(args.video_path)
    generate_video_caption(args.video_path)
    clean_files_and_folders()
    

if __name__ == '__main__':
    main()