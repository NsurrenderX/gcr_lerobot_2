import os
import cv2
import torch
import json
import time
import base64
import requests
import transformers
import numpy as np

from torchvision import transforms
from PIL import Image
from flask import Flask, jsonify, request
from qwen_vl_utils import process_vision_info

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.common.policies.factory import make_policy
from lerobot.common.datasets.transforms import ImageTransforms
from lerobot.common.datasets.lerobot_dataset import MultiSameDataset
from lerobot.common.datasets.rotation_convert import (
    quaternion_to_ortho6d,
    ortho6d_to_euler
)

from torch.utils.data import DataLoader
# from torch.utils.data import Dataloader

def pil2tensor(pil_image, transform_enable=False):
    if transform_enable:
        transform_pipe = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)
        ])
        pil_image = transform_pipe(pil_image)
        pil_image = pil_image.unsqueeze(0)
        return pil_image
    else:
        np_pil_image = np.array(pil_image).astype(np.float32)
        np_pil_image = np_pil_image.transpose((2, 0, 1)) / 255.0
        tensor_image = torch.from_numpy(np_pil_image)
        tensor_image = tensor_image.unsqueeze(0)
        return tensor_image
    

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
    image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
    # Get h,w of image
    # h, w, _ = image_cv2.shape
    # #Crop the image retain the mid area to be a square
    # if h > w:
    #     image_cv2 = image_cv2[int((h-w)/2):int((h+w)/2), :, :]
    # elif w > h:
    #     image_cv2 = image_cv2[:, int((w-h)/2):int((w+h)/2), :]
    return image_cv2

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
    global device, xvla, state_mean, state_std, action_mean, action_std, dim_act
    
    resp = request.get_json()
    fix_state = np.zeros((8))
    item = {
        "dataset_name": ["cup_full_plus"],
        "observation.state": torch.tensor(resp["state"]),
        # "observation.state": torch.tensor(fix_state),
        "task": [resp["task"]],
        "exp_id": resp["exp_id"]
    }
    
    images = resp['images'][0]
    item["primary"] = []
    for img in images:
        image_k4a_1 = decode_b64_image(img)
        image_k4a_1 = image_k4a_1[40:720,200:880,:] 
        image_k4a_1 = Image.fromarray(image_k4a_1).resize((224, 224))
        image_k4a_1.save("/home/v-wenhuitan/pi_0_open/media/obs/xvla_k4a.jpg")
        item["primary"].append(image_k4a_1)
        
    item["observation.images.primary"] = item["primary"][-1]
    item["observation.images.primary"] = pil2tensor(item["observation.images.primary"], transform_enable=True)
    
    # sample_image = np.ones((224,224,3), dtype=np.uint8)
    # sample_image = sample_image * -1
    # sample_image = Image.fromarray(sample_image)
    
    item["observation.images.secondary"] = decode_b64_image(resp['images'][1])
    top_shape = item["observation.images.secondary"].shape
    if top_shape[0] > top_shape[1]:
        # center crop
        item["observation.images.secondary"] = item["observation.images.secondary"][top_shape[0]//2 - top_shape[1]//2:top_shape[0]//2 + top_shape[1]//2,:,:]
    else:
        item["observation.images.secondary"] = item["observation.images.secondary"][:,top_shape[1]//2-top_shape[0]//2:top_shape[1]//2+top_shape[0]//2,:]
    item["observation.images.secondary"] = Image.fromarray(item["observation.images.secondary"]).resize((224,224))
    
    # item["observation.images.secondary"] = sample_image
    item["observation.images.secondary"].save("/home/v-wenhuitan/pi_0_open/media/obs/real_1.jpg")
    item["observation.images.secondary"] = pil2tensor(item["observation.images.secondary"], transform_enable=True)
    
    item["observation.images.wrist"] = decode_b64_image(resp['images'][2])
    wrist_shape = item["observation.images.wrist"].shape
    if wrist_shape[0] > wrist_shape[1]:
        # center crop
        item["observation.images.wrist"] = item["observation.images.wrist"][wrist_shape[0]//2 - wrist_shape[1]//2:wrist_shape[0]//2 + wrist_shape[1]//2,:,:]
    else:
        item["observation.images.wrist"] = item["observation.images.wrist"][:,wrist_shape[1]//2-wrist_shape[0]//2:wrist_shape[1]//2+wrist_shape[0]//2,:]
    item["observation.images.wrist"] = Image.fromarray(item["observation.images.wrist"]).resize((224,224))
    # item["observation.images.wrist"] = sample_image
    item["observation.images.wrist"].save("/home/v-wenhuitan/pi_0_open/media/obs/real_2.jpg")
    item["observation.images.wrist"] = pil2tensor(item["observation.images.wrist"], transform_enable=True)
    
    image_input = torch.cat([item["observation.images.primary"], item["observation.images.secondary"], item["observation.images.wrist"]], dim=0)
    item["image_input"] = image_input
    view_count = image_input.shape[0]
    item['image_mask'] = torch.ones(view_count, dtype=torch.float32)
    item["image_input"] = item["image_input"].unsqueeze(0).to(dtype=torch.float32)
    item["image_mask"] = item["image_mask"].unsqueeze(0).to(dtype=torch.float32)
    del item["observation.images.primary"]
    del item["observation.images.secondary"]
    del item["observation.images.wrist"]
    
    # item["observation.state"] = (item["observation.state"] - state_mean) / (state_std + 1e-8) 
    state = torch.zeros(dim_act, dtype=torch.float32)
    eef_pos = item['observation.state'][:3]
    eef_quat = item['observation.state'][3:7]
    w, x, y, z = eef_quat
    eef_quat = [x, y, z, w]
    eef_quat = torch.tensor(eef_quat)
    eef_orth6d = quaternion_to_ortho6d(eef_quat)
    gripper = item['observation.state'][7:8]
    state[:3] = eef_pos
    state[3:9] = eef_orth6d
    state[9:10] = gripper
    item['observation.state'] = state
    item["observation.state"] = item["observation.state"].unsqueeze(0).to(dtype=torch.float32)
    
    print(f"image dtype: {item['image_input'].dtype}")
    for k,v in item.items():
        if isinstance(v, torch.Tensor):
            item[k] = v.to(device)
    
    actions = xvla.select_action(item)[0] # 1 * 30 * 20
    chunk_size = actions.shape[0]
    # actions = actions[0] # 50 * 32
    action_return = torch.zeros((chunk_size, 7), dtype=torch.float32)
    act_tran = actions[:, :3]
    act_roth6d = actions[:, 3:9]
    act_gripper = actions[:, 9:10]
    act_euler = ortho6d_to_euler(act_roth6d)
    
    action_return[:, :3] = act_tran
    action_return[:, 3:6] = act_euler
    action_return[:, 6:7] = act_gripper
    action_return = action_return.numpy()
    
    print(action_return)
    
    response_dict = {
        "act": action_return.tolist()
    }
    return jsonify(response_dict)
# load qwen2.5vl's vl processor when calling __init__

@parser.wrap()
def start_service(cfg: TrainPipelineConfig):
    
    # path_2_load = "/data_16T/deepseek/pi0pizza/mp_rank_00_model_states.pt"
    path_2_load = "/data_16T/deepseek/xvla_finetune/cupp_1112_fs50/step15000.pt"
    device = "cuda:1"
    
    image_transforms = (ImageTransforms(cfg.dataset.image_transforms))
    seed = cfg.seed
    
    print(f"uni res: {cfg.uni_res}")
    dataset = MultiSameDataset(
        cfg=cfg, 
        image_transforms=image_transforms,
        # seed=seed,
        # data_mix=cfg.data_mix,
        vla2root_json="vla2root.json",
        # vla2root_json="vla2root_bak_single.json"
    )
    
    loader = DataLoader(
        dataset=dataset,
        batch_size=2,
    )
    print("\n"+"-"*20 + "Dataset Summary" + "-"*20)
    data = next(iter(loader))
    for k,v in data.items():
        if isinstance(v, torch.Tensor):
            print(k, v.shape)
        elif isinstance(v, list):
            print(k, len(v), v[0])
        # print(k, v.shape)
    print("-"*40+"\n")
    action_mean = dataset.stats["action"]["mean"][:14]
    dataset.stats["action"]["mean"] = dataset.stats["action"]["mean"][:14]
    action_std = dataset.stats["action"]["std"][:14]
    dataset.stats["action"]["std"] = dataset.stats["action"]["std"][:14]
    print("Action: ", action_mean, action_std)
    # print(action_mean, action_std)
    # action_mean[6] = 0.0
    # action_std[6] = 1.0
    
    state_mean = dataset.stats["observation.state"]["mean"][:15]
    dataset.stats["observation.state"]["mean"] = dataset.stats["observation.state"]["mean"][:15]
    state_std = dataset.stats["observation.state"]["std"][:15]
    dataset.stats["observation.state"]["std"] = dataset.stats["observation.state"]["std"][:15]
    print("State: ", state_mean, state_std)
    # processor = dataset.processor
    dataset.meta.stats['observation.state']['mean'] = np.zeros_like(dataset.meta.stats['observation.state']['mean'])
    dataset.meta.stats['observation.state']['std'] = np.ones_like(dataset.meta.stats['observation.state']['std'])
    # print("meta stats: ", dataset.meta.stats)
    
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
            # print(k)
            if k == "unnormalize_outputs.buffer_action.mean":
                print(k, v)
                print(action_mean, "\n")
            if k == "unnormalize_outputs.buffer_action.std":
                print(k, v)
                print(action_std, "\n")
            if k == "normalize_inputs.buffer_observation_state.mean":
                print(k, v)
            if k == "normalize_inputs.buffer_observation_state.std":
                print(k, v)
            if "awa_model.lm_head" in k or "qwen_expert.lm_head" in k:
                key_to_remove.append(k)
        for k in key_to_remove:
            del model_state_dict[k]
            
        # policy.load_state_dict(model_state_dict["module"], strict=True)
        policy.load_state_dict(model_state_dict, strict=True)
        del model_state_dict
        del key_to_remove
    
    # for params in policy.parameters():
    #     params.data = params.data.bfloat16()
        
    xvla = policy
    xvla.eval()

    xvla.to(device=device)
    
    
    return device, xvla, state_mean, state_std, action_mean, action_std

if __name__ == '__main__':
    device, xvla, state_mean, state_std, action_mean, action_std = start_service()
    dim_act = 20
    # image_pool = {}
    app.run(host='0.0.0.0', port=7778)