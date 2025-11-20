import os
import cv2
import torch
import json
import time
import base64
import requests
import transformers
import numpy as np
import copy

from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from flask import Flask, jsonify, request
from qwen_vl_utils import process_vision_info

from scipy.spatial.transform import Rotation as R

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.common.policies.factory import make_policy
from lerobot.common.datasets.transforms import ImageTransforms
from lerobot.common.datasets.utils import cycle
from lerobot.common.datasets.lerobot_dataset import MultiDatasetforDistTraining, extra_collate_fn

def prepare_images(item: dict):
    vision = {
        "image": [],
        "video": None
    }
    
    all_image_keys = ["primary"]
    present_img_keys = [key for key in all_image_keys if key in item]
    if len(present_img_keys) == 0:
        raise ValueError(
            f"item must contain at least one of the following keys: {all_image_keys}"
        )

    vision["image"].append(item["primary"][-1])
        
    video = item["primary"]
    video_length = len(video)
    for i in range(video_length):
        video[i] = video[i].resize((112, 112))
    
    vision["image"].append(item["secondary"])
    vision["image"].append(item["wrist"])
    default_image = Image.fromarray(np.ones((224, 224, 3), dtype=np.uint8))
    # vision["image"].append(default_image)
    # vision["image"].append(default_image)
    vision["video"] = video[-3:]
    
    return vision

def prepare_language(vision, processor, item):
    def apply_template(text, vision=None):
        message = [
            {"role": "user",
            "content": [
                
            ],}
        ]
        if "video" in vision.keys():
            
            if vision["video"] is not None:
                message[0]["content"].append(
                    {
                        "type": "video",
                        "video": vision["video"],
                    },
                )
            
        for i in range(len(vision["image"])):
            message[0]["content"].append(
                {
                    "type": "image",
                    "image": vision["image"][i],
                }
            )
        message[0]["content"].append({"type": "text", "text": text})
        return processor.apply_chat_template(
                message, tokenize=False, add_generation_prompt=True
            )
    text = item["task"]
    
    template = apply_template(text, vision)
    
    return template

def pad_vector(vector, new_dim):
    """Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)
    """
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
    new_vector[..., :current_dim] = vector
    return new_vector

def prepare_data(item, processor):
    
    vision = prepare_images(item)
    task = prepare_language(vision, processor, item)
    
    text = item["task"]
    
    message = [
        {
            "role": "user",
            "content": []
        }
    ]
        
    video = vision["video"]
    if video is not None:
        message[0]["content"].append(
            {
                "type": "video",
                "video": video,
            },
        )
    for i in range(len(vision["image"])):
        message[0]["content"].append(
            {
                "type": "image",
                "image": vision["image"][i],
            }
        )
    message[0]["content"].append({"type": "text", "text": text})
    
    image_inputs, video_inputs, video_kwargs = process_vision_info(message, return_video_kwargs=True)
        
    inputs = processor(
        text=task,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors = "pt",
        **video_kwargs,
    )
    
    input_ids = getattr(inputs, "input_ids", None)
    attention_mask = getattr(inputs, "attention_mask", None)
    pixel_values = getattr(inputs, "pixel_values", None)
    image_grid_thw = getattr(inputs, "image_grid_thw", None)
    pixel_values_videos = getattr(inputs, "pixel_values_videos", None)
    video_grid_thw = getattr(inputs, "video_grid_thw", None)
    second_per_grid_ts = getattr(inputs, "second_per_grid_ts", None)
    
    return_dict = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "pixel_values_videos": pixel_values_videos,
        "video_grid_thw": video_grid_thw,
        "second_per_grid_ts": second_per_grid_ts,
    }

    return return_dict

def prepare_input(item, processor, device):
    vl_item = prepare_data(item, processor)
    
    item["observation.state"] = pad_vector(item["observation.state"], 32)
    item["observation.state"] = (item["observation.state"] - item["mean"]) / (item["std"] + 1e-8)
    # state = torch.zeros_like(item["observation.state"])
    # state_prefix = torch.ones_like(item["observation.state"])
    # state_device = item["observation.state"].device
    # state_dtype = item["observation.state"].dtype

    # state[:8] = state_prefix[:8]
    # state = state.to(device = state_device, dtype = state_dtype)
    # item["observation.state"] = state
    
    data_dict = {
        "observation.state": item["observation.state"].unsqueeze(0),
        "action": None,
        **vl_item,
    }
    for k,v in data_dict.items():
        if isinstance(v, torch.Tensor):
            data_dict[k] = v.to(device)
        elif isinstance(v, list):
            if len(v) > 0:
                if isinstance(v[0], torch.Tensor):
                    device_list = [v[i].to(device) for i in range(len(v))]
                    data_dict[k] = device_list
    return data_dict
@parser.wrap()
def main(cfg: TrainPipelineConfig):
    
    obs_seq_len = 3
    path_2_load = "/data_16T/deepseek/qwen_flow/161/step10000.pt"
    cfg.policy.qwen_path = "/datassd_1T/qwen25vl/Qwen2.5-VL-3B-Instruct/"
    device = "cuda:0"
    
    image_transforms = (ImageTransforms(cfg.dataset.image_transforms))
    seed = cfg.seed
    dataset = MultiDatasetforDistTraining(
        cfg=cfg, 
        image_transforms=image_transforms,
        seed=seed,
        data_mix=cfg.data_mix,
        vla2root_json="vla2root.json",
    )
    ACT_IDX = [0,1,2,3,4,5,16,17,18,19,20,21,22,33]
    STATE_IDX = [0,1,2,6,7,8,9,16,17,18,19,23,24,25,26,33]
    action_mean = F.pad(dataset.stats["action"]["mean"][ACT_IDX], (0, 32 - dataset.stats["action"]["mean"][ACT_IDX].shape[0]))
    action_std = F.pad(dataset.stats["action"]["std"][ACT_IDX], (0, 32 - dataset.stats["action"]["std"][ACT_IDX].shape[0]))
    
    print("Action Meta: \n", action_mean, action_std)
    
    state_mean = F.pad(dataset.stats["observation.state"]["mean"][STATE_IDX], (0, 32 - dataset.stats["observation.state"]["mean"][STATE_IDX].shape[0]))
    state_std = F.pad(dataset.stats["observation.state"]["std"][STATE_IDX], (0, 32 - dataset.stats["observation.state"]["std"][STATE_IDX].shape[0]))
    
    processor = dataset.processor
    
    policy = make_policy(
        cfg=cfg.policy,
        device="cpu",
        ds_meta=dataset.meta,
        weight_pt_path=cfg.policy.pretrained_path
    )
    
    if path_2_load:
        model_state_dict = torch.load(path_2_load, map_location="cpu")
        model_state_dict = model_state_dict['module']
        key_to_remove = []
        for k, v in model_state_dict.items():
            if "awa_model.lm_head" in k or "qwen_expert.lm_head" in k:
                key_to_remove.append(k)
        for k in key_to_remove:
            del model_state_dict[k]
            
        policy.load_state_dict(model_state_dict, strict=True)
        del model_state_dict
        del key_to_remove
    
    for params in policy.parameters():
        params.data = params.data.bfloat16()
        
    lola = policy
    lola.eval()

    lola.to(device=device)
    
    sim_image = np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8)
    sim_image = Image.fromarray(sim_image)
    
    simulation_data = {
        "observation.state": torch.ones(32).to(dtype=torch.float32),
        "mean": state_mean,
        "std": state_std,
        "task": "Pick up the apple.",
    }
    
    simulation_data['primary'] = [sim_image for _ in range(obs_seq_len)]
    simulation_data['secondary'] = sim_image
    simulation_data['wrist'] = sim_image
    
    input = prepare_input(simulation_data, processor, device)
    input["action.mean"] = action_mean
    input["action.std"] = action_std
    actions = lola.infer(input).tolist()
    
    actions = np.array([row[:14] for row in actions[0]])
    GRIPPER_IDX = [6, 13]
    actions[:, GRIPPER_IDX] = 1 / (1 + np.exp(-actions[:, GRIPPER_IDX]))
    
    print(actions.shape)
    response_dict = {
        "act": actions
    }
    
    return response_dict

if __name__ == "__main__":
    main()