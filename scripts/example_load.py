from lerobot.common.policies.factory import make_policy
from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
import torch
import numpy as np

"""
python example_load.py --policy.type=pi05 --env.type=msra-ee
"""
@parser.wrap()
def main(cfg: EvalPipelineConfig):
    print("Start service")
    weight_path = "/data/lola/global_step60000/mp_rank_00_model_states.pt"
    # weight_path = cfg.weight_path

    device = "cuda:0" if cfg.device == "cuda" else cfg.device
    cfg.policy.add_new_tokens = True
    # cfg.policy.use_new_tokens = True
    model = make_policy(
        cfg=cfg.policy,
        device=device,
        env_cfg=cfg.env,
        weight_pt_path=weight_path,
    )
    if False:
        weights = torch.load(weight_path, map_location="cpu")["module"]
        missing_keys, unexpected_keys = model.load_state_dict(weights, strict=False)
        print(f"missing:{missing_keys} unexpected:{unexpected_keys}")
        print(f"Loaded weights from {weight_path}")
    # Use the resolved device for model and tensors
    model.to(device)
    # input data
    task_description = "pick up the cup"
    cam_high_img_np = np.random.randint(0, 256, size=(480, 840, 3), dtype=np.uint8)
    cam_left_img_np = np.random.randint(0, 256, size=(480, 840, 3), dtype=np.uint8)
    cam_right_img_np = np.random.randint(0, 256, size=(480, 840, 3), dtype=np.uint8)
    state_np = np.random.rand(16) # dual arm should be 16. The state in training is x,y,z,quat, gripper for both hand
    # image range should be 0-1
    cam_high_image = torch.from_numpy(cam_high_img_np / 255).permute(2, 0, 1).unsqueeze(0).to(device).float()
    cam_left_image = torch.from_numpy(cam_left_img_np / 255).permute(2, 0, 1).unsqueeze(0).to(device).float()
    cam_right_image = torch.from_numpy(cam_right_img_np / 255).permute(2, 0, 1).unsqueeze(0).to(device).float()

    state = torch.from_numpy(state_np).unsqueeze(0).to(device).float()
    observation = {
        "observation.images.cam_high": cam_high_image,
        "observation.images.cam_left_wrist": cam_left_image,
        "observation.images.cam_right_wrist": cam_right_image,
        "observation.state": state,
        "task": [str(task_description)]
    }
    # Query model to get action
    with torch.inference_mode():
        actions = model.select_action(observation) # B chunk_size action_dim
        print(f"Action type: {type(actions)}")
        print(f"Action shape: {actions.shape}")
        print(f"First action values: {actions[0, :3]}")

if __name__ == "__main__":
    main()
