import os
import cv2
import torch
import json
import time
import base64
import requests
import math
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

def get_predict_action(batch, model, devicetype):
    # for k, v in batch.items():
    #     if isinstance(v, torch.Tensor):
    #         batch[k] = v.to(devicetype)
    # batch["action.mean"] = action_mean
    # batch["action.std"] = action_std
    
    actions = model.infer(batch)
    
    predictd_actions = []
    for i in range(actions.shape[0]):
        predictd_action = [action[:6] for action in actions[i]]
        predictd_actions.append(predictd_action)

    predictd_actions = np.array(predictd_actions)
    
    return predictd_actions

def act_delta(actions, batch):
    assert isinstance(actions, np.ndarray) , "actions must be a numpy array"
    # "observation.state"
    states = batch["observation.state"].cpu()
    state_mean = batch["state.mean"].cpu()
    state_std = batch["state.std"].cpu()
    states = states*(state_std+1e-8) + state_mean
    states = states.cpu().numpy()
    # predict state
    p_states = []
    a_states = []
    batch_size = states.shape[0]
    # normed_action = copy.deepcopy( batch["action"].cpu().numpy())
    # batch["action"] = batch["action"].cpu().to(dtype=torch.float64)*(batch["action.std"].cpu().to(dtype=torch.float64) + 1e-8) + batch["action.mean"].cpu().to(dtype=torch.float64)
        
    for i in range(batch_size):
        state = states[i]
        p_state_i = []
        p_state = np.zeros(6)
        p_state[:3] = state[:3]
        w, x, y, z = state[3:7]
        r_mat = R.from_quat([x, y, z, w]).as_matrix()
        for j in range(actions.shape[1]):
            p_state[:3] += actions[i, j, :3]
            r_mat = r_mat @ R.from_euler('xyz', actions[i, j, 3:]).as_matrix()
            p_state[3:] = R.from_matrix(r_mat).as_euler('xyz', degrees=False)
            predict_state = copy.deepcopy(p_state)
            p_state_i.append(predict_state)
            del predict_state
            
        predict_state_i = copy.deepcopy(p_state_i)  
        p_states.append(predict_state_i)
        del predict_state_i
        a_state_i = []
        a_state = np.zeros(6)
        a_state[:3] = state[:3]
        w, x, y, z = state[3:7]
        r_mat = R.from_quat([x, y, z, w]).as_matrix()
        # actual_action = copy.deepcopy(batch['action']).cpu()[i]
        # action_std = batch["action.std"].cpu()
        # action_mean = batch["action.mean"].cpu()
        # normed_action = actual_action*(action_std + 1e-8) + action_mean
        # denormed_action = (normed_action-action_mean)/(action_std+1e-8)
        
        actual_act = batch['action'][i].cpu().numpy()
        for j in range(actual_act.shape[0]):
            a_state[:3] += actual_act[j, :3]
            r_mat = r_mat @ R.from_euler('xyz', actual_act[j, 3:6]).as_matrix()
            a_state[3:] = R.from_matrix(r_mat).as_euler('xyz', degrees=False)
            actual_state = copy.deepcopy(a_state)
            a_state_i.append(actual_state)
            del actual_state
        actual_state_i = copy.deepcopy(a_state_i)
        a_states.append(actual_state_i)
        del actual_state_i
    # print(f"predict state: {p_state_i[:5]}")
    # print(f"actual state: {a_state_i[:5]}")
    # print(f"predict action: {actions[-1][0][:6]}")
    # print(f"Ori action: {batch['ori_action'].cpu().numpy()[-1][0][:6]}")
    # print(f"Denormed action: {batch['action'].cpu().numpy()[-1][0][:6]}")
    # print(f"Normed action: {normed_action[-1][0][:6]}")
    
    # print(f"Diff after denorm: {denormed_action - actual_action}\nmean: {torch.mean(denormed_action - actual_action)}, std: {torch.std(denormed_action - actual_action)}")
    # batch x sequence_length x 6
    return np.array(p_states), np.array(a_states)
            
def get_predict_error(p_states, a_states):
    error = p_states - a_states
    init_error = error[:, 0, :]
    step_5_error = error[:, 4, :]
    step_10_error = error[:, 9, :]
    step_15_error = error[:, 14, :]
    step_20_error = error[:, 19, :]
    step_25_error = error[:, 24, :]
    print(np.mean(init_error-step_15_error))
    return init_error, step_5_error, step_10_error, step_15_error, step_20_error, step_25_error

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

def dict_json_serlizable(d:dict):
    #convert all numpy ndarray in dict to list recursively
    for k,v in d.items():
        if isinstance(v, np.ndarray):
            d[k] = v.tolist()
            print("converted numpy ndarray:{} to list".format(k))
        elif isinstance(v, dict):
            dict_json_serlizable(v)
        elif isinstance(v, list):
            for i in range(len(v)):
                if isinstance(v[i], np.ndarray):
                    v[i] = v[i].tolist()
                    print("converted numpy ndarray: {} to Obj.list".format(k))
    return d

def numpy_to_json(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist() # Convert NumPy array to a list
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

def decode_b64_image(b64image):
    # Decode the base64 image string
    str_decode = base64.b64decode(b64image)
    np_image = np.frombuffer(str_decode, np.uint8)
    image_cv2 = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
    # Get h,w of image
    h, w, _ = image_cv2.shape
    image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
    #Crop the image retain the mid area to be a square
    # if h > w:
    #     image_cv2 = image_cv2[int((h-w)/2):int((h+w)/2), :, :]
    # elif w > h:
    #     image_cv2 = image_cv2[:, int((w-h)/2):int((w+h)/2), :]
    return image_cv2

def prepare_images(item: dict):
    vision = {
        "image": [],
        "video": None
    }
    
    assert "exp_id" in item, "item must contain exp_id"
    
    all_image_keys = ["primary"]
    present_img_keys = [key for key in all_image_keys if key in item]
    if len(present_img_keys) == 0:
        raise ValueError(
            f"item must contain at least one of the following keys: {all_image_keys}"
        )
    if item["exp_id"] in image_pool:
        image_pool[item["exp_id"]].append(item["primary"])
    else:
        image_pool[item["exp_id"]] = [item["primary"]]
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

def prepare_input(item, processor):
    vl_item = prepare_data(item, processor)
    
    item["observation.state"] = pad_vector(item["observation.state"], 32)
    item["observation.state"][7] = 0 if item["observation.state"][7] < 0.06 else 1
    item["observation.state"] = (item["observation.state"] - item["mean"]) / (item["std"] + 1e-8)
    state = torch.zeros_like(item["observation.state"])
    state_prefix = torch.ones_like(item["observation.state"])
    state_device = item["observation.state"].device
    state_dtype = item["observation.state"].dtype
    # state[:8] = item["observation.state"][:8]
    state[:8] = state_prefix[:8]
    state = state.to(device = state_device, dtype = state_dtype)
    item["observation.state"] = state
    
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
    
    print("\n-------------------------------------------\n")
    print(f"input_ids: {data_dict['input_ids'].shape}")
    print(f"attention_mask: {data_dict['attention_mask'].shape}")
    print(f"pixel_values: {data_dict['pixel_values'].shape}")
    print(f"image_grid_thw: {data_dict['image_grid_thw']}")
    print(f"pixel_values_videos: {data_dict['pixel_values_videos'].shape}")
    print(f"video_grid_thw: {data_dict['video_grid_thw']}")
    print(f"second_per_grid_ts: {data_dict['second_per_grid_ts']}")

    return data_dict

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Cuda realtime vla model inference service!'

@app.route('/predict', methods=['POST'])
def predict():
    # task: str
    # image(s): in list type of base64 encoded string
    # exp_id: str
    # current state: list[float]

    # process the images into CV2.image or IMAGE.image
    # rerun the prepare_data process to get the desired input
    # global resource pool: image list stored in different exp_ids
    # global resource pool: processor
    global device, image_pool, processor, halo, state_mean, state_std, action_mean, action_std
    
    resp = request.get_json()
    
    item = {
        "observation.state": torch.tensor(resp["state"]),
        "mean": state_mean,
        "std": state_std,
        "task": resp["task"],
        "exp_id": resp["exp_id"]
    }
    
    images = resp['images'][0]
    item["primary"] = []
    for img in images:
        image_k4a_1 = decode_b64_image(img)
        image_k4a_1 = image_k4a_1[40:720,200:880,:] 
        image_k4a_1 = Image.fromarray(image_k4a_1).resize((224,224))
        # save image for visulization
        image_k4a_1.save("/home/v-wenhuitan/pi_0_open/media/obs/k4a.jpg")
        item["primary"].append(image_k4a_1)
    sample_image_array = np.ones((224, 224, 3), dtype=np.uint8)
    item["secondary"] = decode_b64_image(resp['images'][1])
    sec_shape = item["secondary"].shape
    if sec_shape[0] > sec_shape[1]:
        # center crop
        item["secondary"] = item["secondary"][sec_shape[0]//2 - sec_shape[1]//2:sec_shape[0]//2 + sec_shape[1]//2,:,:]
    else:
        item["secondary"] = item["secondary"][:,sec_shape[1]//2-sec_shape[0]//2:sec_shape[1]//2+sec_shape[0]//2,:]
    # item["secondary"] = item["secondary"][40:720,200:880,:]
    item["secondary"] = Image.fromarray(item["secondary"]).resize((224,224))
    item["secondary"].save("/home/v-wenhuitan/pi_0_open/media/obs/real_1.jpg")
    item['secondary'] = Image.fromarray(sample_image_array)
    
    item["wrist"] = decode_b64_image(resp['images'][2])
    wrist_shape = item["wrist"].shape
    print(wrist_shape)
    # item["wrist"].save("/home/v-wenhuitan/pi_0_open/media/obs/real_2_ori.jpg")
    if wrist_shape[0] > wrist_shape[1]:
        # center crop
        item["wrist"] = item["wrist"][wrist_shape[0]//2 - wrist_shape[1]//2:wrist_shape[0]//2 + wrist_shape[1]//2,:,:]
    else:
        item["wrist"] = item["wrist"][:,wrist_shape[1]//2-wrist_shape[0]//2:wrist_shape[1]//2+wrist_shape[0]//2,:]
    
    item["wrist"] = Image.fromarray(item["wrist"]).resize((224,224))
    item["wrist"].save("/home/v-wenhuitan/pi_0_open/media/obs/real_2.jpg")
    item['wrist'] = Image.fromarray(sample_image_array)
    
    input = prepare_input(item, processor)
    input["action.mean"] = action_mean
    input["action.std"] = action_std
    
    actions = halo.infer(input).tolist() # 1 * 50 *32
    chunk = len(actions[0])
    for i in range(chunk):
        actions[0][i][6] = 1 / (1+ math.exp(-actions[0][i][6]))
    # actions = actions[0] # 50 * 32
    action_list = [row[:7] for row in actions[0]] # 50 * 7 eef pose
    # actions = [row[6:14] for row in actions[0]] # 50 * 7 joint
    for i in range(10):
        row = action_list[i]
        print(f" ".join(f"{num:4f}" for num in row))
        # print(actions[i])
    # print()
    
    response_dict = {
        "act": action_list
    }
    return jsonify(response_dict)

@app.route('/exppredict', methods=['POST'])
def exp_predict():
    # task: str
    # image(s): in list type of base64 encoded string
    # exp_id: str
    # current state: list[float]

    # process the images into CV2.image or IMAGE.image
    # rerun the prepare_data process to get the desired input
    # global resource pool: image list stored in different exp_ids
    # global resource pool: processor
    global device, image_pool, processor, halo, state_mean, state_std, action_mean, action_std
    
    resp = request.get_json()
    
    resp["state"] = torch.tensor(resp["state"])
    
    item = {
        "observation.state": torch.ones_like(resp["state"]).to(dtype=torch.float32),
        "mean": state_mean,
        "std": state_std,
        "task": resp["task"],
        "exp_id": resp["exp_id"]
    }
    
    images = resp['images'][0]
    item["primary"] = []
    for img in images:
        image_k4a_1 = decode_b64_image(img)
        image_k4a_1 = image_k4a_1[40:720,200:880,:] 
        image_k4a_1 = Image.fromarray(image_k4a_1).resize((224,224))
        # save image for visulization
        
        item["primary"].append(image_k4a_1)
    image_k4a_1 = Image.open("/home/v-wenhuitan/pi_0_open/media/primary-3969-9.jpg").resize((224,224))
    image_k4a_1.save("/home/v-wenhuitan/pi_0_open/media/obs/k4a.jpg")
    item["primary"].append(image_k4a_1)
    item["secondary"] = decode_b64_image(resp['images'][1])
    # item["secondary"] = item["secondary"][40:720,200:880,:]
    item["secondary"] = Image.fromarray(item["secondary"]).resize((224,224))
    item["secondary"].save("/home/v-wenhuitan/pi_0_open/media/obs/real_1.jpg")
    item["secondary"] = Image.open("/home/v-wenhuitan/pi_0_open/media/secondary-3969.jpg").resize((224,224))
    item["secondary"].save("/home/v-wenhuitan/pi_0_open/media/obs/real_1.jpg")
    
    item["wrist"] = decode_b64_image(resp['images'][2])
    wrist_shape = item["wrist"].shape
    
    if wrist_shape[0] > wrist_shape[1]:
        # center crop
        item["wrist"] = item["wrist"][wrist_shape[0]//2 - wrist_shape[1]//2:wrist_shape[0]//2 + wrist_shape[1]//2,:,:]
    else:
        item["wrist"] = item["wrist"][:,wrist_shape[1]//2-wrist_shape[0]//2:wrist_shape[1]//2+wrist_shape[0]//2,:]
    item["wrist"] = Image.fromarray(item["wrist"]).resize((224,224))
    item["wrist"].save("/home/v-wenhuitan/pi_0_open/media/obs/real_2.jpg")
    
    input = prepare_input(item, processor)
    input["action.mean"] = action_mean
    input["action.std"] = action_std
    
    actions = halo.infer(input).tolist() # 1 * 50 *32
    # actions = actions[0] # 50 * 32
    actions = [row[:14] for row in actions[0]] # 50 * 7 eef pose
    # actions = [row[6:14] for row in actions[0]] # 50 * 7 joint
    print(actions[:5])
    
    response_dict = {
        "act": actions
    }
    return jsonify(response_dict)
# load qwen2.5vl's vl processor when calling __init__

@app.route('/modelbench', methods=['GET'])
def modelbench():
    global action_mean, action_std, device, loader_cycler, halo, state_mean, state_std
    batch = next(loader_cycler)
    gap = 5
    for i in range(gap):
        batch = next(loader_cycler)
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
        # print(k)
    batch["action.mean"] = action_mean
    batch["action.std"] = action_std
    
    actions = halo.infer(batch).tolist() # 1 * 50 *32
    actions = actions[0] # 50 * 32
    actions = [row[:7] for row in actions] # 50 * 7 eef pose
    
    print(actions[:5])
    
    denorm_state = batch["observation.state"][0].cpu()*(state_std+1e-8) + state_mean
    
    response_dict = {
        "act": actions,
        "state": batch["observation.state"][0].cpu().numpy().tolist()
    }
    
    return jsonify(response_dict)

@parser.wrap()
def start_service(cfg: TrainPipelineConfig):
    
    path_2_load = "/data_16T/deepseek/halo/cup_pp/step6000.pt"
    cfg.policy.qwen_path = "/datassd_1T/qwen25vl/Qwen2.5-VL-7B-Instruct/"
    device = "cuda:1"
    
    model_bench = False
    
    image_transforms = (ImageTransforms(cfg.dataset.image_transforms))
    seed = cfg.seed
    dataset = MultiDatasetforDistTraining(
        cfg=cfg, 
        image_transforms=image_transforms,
        seed=seed,
        data_mix=cfg.data_mix,
        vla2root_json="vla2root.json",
        # vla2root_json="vla2root_bak_single.json"
    )
    ACT_IDX = [0,1,2,3,4,5]
    STATE_IDX = [0,1,2,6,7,8,9]
    GRIPPER_IDX = 6
    action_mean = F.pad(dataset.stats["action"]["mean"][ACT_IDX], (0, 32 - dataset.stats["action"]["mean"][ACT_IDX].shape[0]))
    action_std = F.pad(dataset.stats["action"]["std"][ACT_IDX], (0, 32 - dataset.stats["action"]["std"][ACT_IDX].shape[0]))
    action_mean[GRIPPER_IDX] = 0.0
    action_std[GRIPPER_IDX] = 1.0
    
    state_mean = F.pad(dataset.stats["observation.state"]["mean"][STATE_IDX], (0, 32 - dataset.stats["observation.state"]["mean"][STATE_IDX].shape[0]))
    state_std = F.pad(dataset.stats["observation.state"]["std"][STATE_IDX], (0, 32 - dataset.stats["observation.state"]["std"][STATE_IDX].shape[0]))
    state_mean[GRIPPER_IDX+1] = 0.0
    state_std[GRIPPER_IDX+1] = 1.0
    
    print("Action Meta: \n", action_mean, action_std)
    # action_mean[6] = 0.0
    # action_std[6] = 1.0
    

    processor = dataset.processor
    
    policy = make_policy(
        cfg=cfg.policy,
        device="cpu",
        ds_meta=dataset.meta,
        weight_pt_path=cfg.policy.pretrained_path
    )
    
    if path_2_load:
        model_state_dict = torch.load(path_2_load, map_location="cpu")
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
        
    halo = policy
    halo.eval()

    halo.to(device=device)
    
    init_errors = []
    step_5_errors = []
    step_10_errors = []
    step_15_errors = []
    step_20_errors = []
    step_25_errors = []
    
    if model_bench:
        benchloader = torch.utils.data.DataLoader(dataset, batch_size=20, shuffle=True, collate_fn=extra_collate_fn, num_workers=4)
        loader_cycler = cycle(benchloader)
        for i in tqdm(range(200)):
            batch = next(loader_cycler)
            print(batch['source'])
            # print(batch['action'][0][0])# bacth x chunk size x action dim
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            batch["action.mean"] = action_mean.cpu()
            batch["action.std"] = action_std.cpu()
            batch["state.mean"] = state_mean.cpu()
            batch["state.std"] = state_std.cpu()
            p_act = get_predict_action(batch, halo, device)
            # print(batch['action'][0][0])
            
            predict_state, actual_state = act_delta(p_act, batch)
            # print(batch['action'][0][0])
            init_error, step_5_error, step_10_error, step_15_error, step_20_error, step_25_error = get_predict_error(predict_state, actual_state)
            # print(step_20_error.shape)
            init_errors.append(init_error)
            step_5_errors.append(step_5_error)
            step_10_errors.append(step_10_error)
            step_15_errors.append(step_15_error)
            step_20_errors.append(step_20_error)
            step_25_errors.append(step_25_error)
    
        init_errors = np.array(init_errors)
        step_5_errors = np.array(step_5_errors)
        step_10_errors = np.array(step_10_errors)
        step_15_errors = np.array(step_15_errors)
        step_20_errors = np.array(step_20_errors)
        step_25_errors = np.array(step_25_errors)
        
        shape_error = init_errors.shape # 10000 x 6
        init_errors = init_errors.reshape(-1, shape_error[-1])
        step_5_errors = step_5_errors.reshape(-1, shape_error[-1])
        step_10_errors = step_10_errors.reshape(-1, shape_error[-1])
        step_15_errors = step_15_errors.reshape(-1, shape_error[-1])
        step_20_errors = step_20_errors.reshape(-1, shape_error[-1])
        step_25_errors = step_25_errors.reshape(-1, shape_error[-1])
        
        print(f"Each dim mean: \nInit:{np.mean(np.abs(init_errors), axis=0)}\nStep5:{np.mean(np.abs(step_5_errors), axis=0)}\nStep10:{np.mean(np.abs(step_10_errors), axis=0)}\nStep15:{np.mean(np.abs(step_15_errors), axis=0)}\nStep20:{np.mean(np.abs(step_20_errors), axis=0)}\nStep25:{np.mean(np.abs(step_25_errors), axis=0)}")
        print(f"Each dim std: \nInit:{np.std(init_errors, axis=0)}\nStep5:{np.std(step_5_errors, axis=0)}\nStep10:{np.std(step_10_errors, axis=0)}\nStep15:{np.std(step_15_errors, axis=0)}\nStep20:{np.std(step_20_errors, axis=0)}\nStep25:{np.std(step_25_errors, axis=0)}")
        print(f"Each dim Max: \nInit:{np.max(np.abs(init_errors), axis=0)}\nStep5:{np.max(np.abs(step_5_errors), axis=0)}\nStep10:{np.max(np.abs(step_10_errors), axis=0)}\nStep15:{np.max(np.abs(step_15_errors), axis=0)}\nStep20:{np.max(np.abs(step_20_errors), axis=0)}\nStep25:{np.max(np.abs(step_25_errors), axis=0)}")
    
    return device, halo, processor, state_mean, state_std, action_mean, action_std, dataset

if __name__ == '__main__':
    device, halo, processor, state_mean, state_std, action_mean, action_std, dataset = start_service()
    image_pool = {}
    benchloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=extra_collate_fn, num_workers=2)
    loader_cycler = cycle(benchloader)
    
    app.run(host='0.0.0.0', port=7777)