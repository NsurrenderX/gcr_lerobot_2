from pathlib import Path
import json
#import draccus
import torch
import socket
import struct
import numpy as np
from PIL import Image
import io

from lola_demo import load_lola_model, OBS_SEQ_LEN, generate_actions
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from math_utils import (
    aloha_calculate_forward_kinematics_quaternion, 
    calculate_forward_kinematics_rpy, 
    ee_to_joints
)

class InferenceServer:
    def __init__(
        self,
        policy,
        host: str = "0.0.0.0",
        port: int = 5000,
    ) -> None:

        self.image_size = 224
        self.model = policy
        self.host = host
        self.port = port
        self.prev_task = None

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    def _recv_exactly(self, conn: socket.socket, n: int) -> bytes:
        data = b""
        while len(data) < n:
            chunk = conn.recv(n - len(data))
            if not chunk:
                raise ConnectionError("Connection closed while receiving data")
            data += chunk
        return data
    
    def receive_observation(self, conn: socket.socket) -> dict:
        """Receive one observation bundle from client using defined protocol."""
        header = self._recv_exactly(conn, 4)
        msg_size = struct.unpack("!I", header)[0]
        raw = self._recv_exactly(conn, msg_size)
        newline_idx = raw.index(b"\n")
        meta_json = raw[:newline_idx].decode("utf-8")
        meta = json.loads(meta_json)
        offset = newline_idx + 1

        eef_state = aloha_calculate_forward_kinematics_quaternion(meta["qpos"])
        obs = {
            "observation.state": torch.tensor(eef_state, dtype=torch.float32),
            "task": meta.get("language", ""),
        }

        print(f"Received CONVERTED observation state: {eef_state}")
        print(f"Current task: {obs['task']}")
        
        cam_remapping = {
            "cam_high": "primary",
            "cam_left_wrist": "secondary",
            "cam_right_wrist": "wrist",
        }

        # --- Decode cam_high sequence sent by the client ---
        cam_high_seq_len = meta["cam_high_seq_len"]
        cam_high_sizes = meta["cam_high_sizes"]

        primary_frames = []
        for img_size in cam_high_sizes:
            img_bytes = raw[offset: offset + img_size]
            offset += img_size
            img = Image.open(io.BytesIO(img_bytes))
            resized = img.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
            primary_frames.append(resized)

        if len(primary_frames) != cam_high_seq_len:
            print(
                f"Warning: cam_high_seq_len ({cam_high_seq_len}) "
                f"!= number of decoded frames ({len(primary_frames)})"
            )

        obs[cam_remapping["cam_high"]] = primary_frames

        # --- Decode single-frame wrist cameras (order must match client) ---
        for key in ["cam_left_wrist", "cam_right_wrist"]:
            size_key = f"{key}_size"
            img_size = meta[size_key]
            img_bytes = raw[offset: offset + img_size]
            offset += img_size
            img = Image.open(io.BytesIO(img_bytes))
            resized = img.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
            obs[cam_remapping[key]] = resized

        return obs

    def send_actions(self, conn: socket.socket, actions: np.ndarray):
        flat = actions.astype(np.float32).tobytes()
        header = struct.pack("!I", len(flat))
        conn.sendall(header)
        conn.sendall(flat)

    def run(self):
        self.socket.bind((self.host, self.port))
        self.socket.listen(1)
        print(f"Inference server listening on {self.host}:{self.port}")
        try:
            while True:
                print("Waiting for client connection...")
                conn, addr = self.socket.accept()
                print(f"Client connected: {addr}")
                try:
                    i = 0
                    joint_actions_old = None
                    while True:
                        i += 1
                        print("Awaiting observation message...")
                        raw_obs = self.receive_observation(conn)
                
                        actions = generate_actions(self.model, raw_obs)
                        for j in range(10):
                            print(f"STEP {i}: EEF ACTION {j}: {actions[j,:]}")
                        joint_actions = ee_to_joints(actions, joint_actions_old)
                        joint_actions_old = joint_actions[-1, :].copy()
                        self.send_actions(conn, joint_actions)
                        print("Action chunk sent\n\n")
                except ConnectionError as e:
                    print(f"Connection error: {e}")
                except Exception as e:
                    print(f"Error during request handling: {e}", exc_info=True)
                finally:
                    conn.close()
                    print("Client disconnected")
        except KeyboardInterrupt:
            print("Shutdown requested (KeyboardInterrupt)")
        finally:
            self.socket.close()
            print("Socket closed; server terminated")



@parser.wrap()
def main(cfg: TrainPipelineConfig):
    policy = load_lola_model(cfg)
    print("Loaded Lola policy successfully.")
    env = InferenceServer(policy)
    env.run()


if __name__ == "__main__":
    main()
    

