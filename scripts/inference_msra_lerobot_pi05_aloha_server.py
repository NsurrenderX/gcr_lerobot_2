#!/usr/bin/env python3

"""Inference server for MSRA Pi05 policies running on the ALOHA robot."""

from __future__ import annotations

import argparse
import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

import draccus
import numpy as np
import torch
from PIL import Image
from pytorch_lightning import seed_everything

from lerobot.common.policies.factory import make_policy
from lerobot.configs.eval import EvalPipelineConfig

from inference_server_base import ServerConfig, InferenceServerBase
from math_utils import calculate_forward_kinematics_quaternion, calculate_inverse_kinematics


logger = logging.getLogger(__name__)


try:  # Pillow>=10 uses the Resampling enum
	RESIZE_MODE = Image.Resampling.BILINEAR
except AttributeError:  # pragma: no cover - Pillow<10 fallback
	RESIZE_MODE = Image.BILINEAR  # type: ignore[attr-defined]


DEFAULT_CHECKPOINT = \
	"/data/lola/global_step60000/mp_rank_00_model_states.pt"


@dataclass
class MsraPi05ServerConfig(ServerConfig):
	checkpoint_path: str = DEFAULT_CHECKPOINT
	chunk_size: int = 50
	dtype: str = "bfloat16"
	seed: int = 0
	resize_width: int = 224
	resize_height: int = 224
	add_new_tokens: bool = True
	language_fallback: str = ""


@dataclass
class MsraPi05RuntimeConfig(MsraPi05ServerConfig):
	eval_cfg: EvalPipelineConfig | None = field(default=None, repr=False)


class MsraPi05AlohaServer(InferenceServerBase):
	"""Inference server that consumes joint observations and outputs joint actions."""

	def __init__(self, config: MsraPi05RuntimeConfig):
		self.runtime_cfg = config
		self.eval_cfg = config.eval_cfg
		if self.eval_cfg is None:
			raise ValueError("EvalPipelineConfig must be provided via runtime config.")
		self.checkpoint_path = config.checkpoint_path
		self.chunk_size = config.chunk_size
		# Actions always express Euler orientation (xyz + rpy + gripper)
		self._action_orientation_dim = 3
		self._arm_action_dim = 3 + self._action_orientation_dim + 1  # 7
		self._action_dim = self._arm_action_dim * 2  # 14
		self.image_size = (config.resize_width, config.resize_height)
		self.dtype = self._resolve_dtype(config.dtype)
		self.seed = config.seed
		self.last_joint_positions: np.ndarray | None = None
		self._ik_seed_left: List[float] | None = None
		self._ik_seed_right: List[float] | None = None
		super().__init__(config)

	def _resolve_dtype(self, dtype_name: str) -> torch.dtype:
		dtype_name = (dtype_name or "").lower()
		if dtype_name in {"bfloat16", "bf16"}:
			if torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)():
				return torch.bfloat16
			logger.warning("bfloat16 requested but unsupported on this device, falling back to float32.")
			return torch.float32
		if dtype_name in {"float16", "fp16"}:
			return torch.float16
		return torch.float32

	def build_model(self):
		seed_everything(self.seed, workers=True)
		eval_cfg = copy.deepcopy(self.eval_cfg)
		if eval_cfg.policy is None:
			raise ValueError("Eval config is missing a policy definition (use --policy.* flags).")
		if self.runtime_cfg.add_new_tokens and hasattr(eval_cfg.policy, "add_new_tokens"):
			eval_cfg.policy.add_new_tokens = True
		logger.info("Building Pi05 policy (%s) for env %s", eval_cfg.policy.type, eval_cfg.env.type)
		model = make_policy(
			cfg=eval_cfg.policy,
			device=self.device,
			env_cfg=eval_cfg.env,
			weight_pt_path=self.checkpoint_path,
		)
		model = model.to(device=self.device, dtype=self.dtype)
		model.eval()
		action_dim = model.config.action_feature.shape[0]
		logger.info(
			"Model ready: dtype=%s chunk_size=%s action_dim=%s",
			next(model.parameters()).dtype,
			model.config.chunk_size,
			action_dim,
		)
		return model

	def process_observation(self, raw_obs: Dict[str, Any]) -> Dict[str, Any]:
		observation: Dict[str, Any] = {}
		for key in self.image_keys:
			img = raw_obs.get(key)
			if img is None:
				raise ValueError(f"Missing image '{key}' in observation.")
			if isinstance(img, np.ndarray):
				img = Image.fromarray(img)
			img = img.convert("RGB").resize(self.image_size, RESIZE_MODE)
			arr = np.asarray(img, dtype=np.float32) / 255.0
			tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
			observation[f"observation.images.{key}"] = tensor.to(self.device, dtype=self.dtype)

		qpos = raw_obs.get("qpos")
		if qpos is None:
			raise ValueError("Observation is missing 'qpos'.")
		if not isinstance(qpos, np.ndarray):
			qpos = np.asarray(qpos, dtype=np.float32)
		if qpos.shape[0] != 14:
			raise ValueError(f"Expected 14 joint positions, received shape {qpos.shape}.")
		self.last_joint_positions = qpos.astype(np.float32)
		ee_state = self._joints_to_ee_state(qpos)
		observation["observation.state"] = (
			torch.from_numpy(ee_state).unsqueeze(0).to(self.device, dtype=self.dtype)
		)

		language = raw_obs.get("language") or self.runtime_cfg.language_fallback
		observation["task"] = [str(language)]
		return observation

	def generate_actions(self, processed_obs: Dict[str, Any]) -> np.ndarray:
		logger.info("Running policy inference...")
		with torch.inference_mode():
			ee_actions = self.model.select_action(processed_obs)
		if ee_actions.ndim != 3:
			raise ValueError(f"Policy returned tensor with shape {tuple(ee_actions.shape)}")
		ee_actions = ee_actions.squeeze(0).to(torch.float32).cpu().numpy()
		logger.info(
			"Received EE chunk: shape=%s range=[%.4f, %.4f]",
			ee_actions.shape,
			ee_actions.min().item() if ee_actions.size else 0.0,
			ee_actions.max().item() if ee_actions.size else 0.0,
		)
		joint_actions = self._ee_to_joint_actions(ee_actions, seed_joints=self.last_joint_positions)
		return joint_actions

	def _joints_to_ee_state(self, joint_positions: np.ndarray) -> np.ndarray:
		left = joint_positions[:7]
		right = joint_positions[7:]
		left_pose = calculate_forward_kinematics_quaternion(left[:6])
		right_pose = calculate_forward_kinematics_quaternion(right[:6])
		left_state = self._format_obs_entry(left_pose, left[6])
		right_state = self._format_obs_entry(right_pose, right[6])
		return np.concatenate([left_state, right_state]).astype(np.float32)

	def _format_obs_entry(self, pose: np.ndarray, gripper: float) -> np.ndarray:
		pos = pose[:3].astype(np.float32)
		quat = pose[3:].astype(np.float32)
		return np.concatenate([pos, quat, np.array([gripper], dtype=np.float32)])

	def _ee_to_joint_actions(
		self,
		ee_actions: np.ndarray,
		seed_joints: np.ndarray | None = None,
	) -> np.ndarray:
		if ee_actions.shape[1] != self._action_dim:
			raise ValueError(
				f"Expected action dimension {self._action_dim}, received {ee_actions.shape[1]}"
			)
		chunk_size = ee_actions.shape[0]
		joint_actions = np.zeros((chunk_size, 14), dtype=np.float32)

		if self._ik_seed_left is None:
			if seed_joints is not None and seed_joints.shape[0] == 14:
				self._ik_seed_left = seed_joints[:6].tolist()
			else:
				self._ik_seed_left = [0.0, -0.96, 1.16, 1.0, -0.3, 0.0]
		if self._ik_seed_right is None:
			if seed_joints is not None and seed_joints.shape[0] == 14:
				self._ik_seed_right = seed_joints[7:13].tolist()
			else:
				self._ik_seed_right = [0.0, -0.96, 1.16, 1.0, -0.3, 0.0]

		left_seed = self._ik_seed_left
		right_seed = self._ik_seed_right

		last_left = np.array(left_seed, dtype=np.float32)
		last_right = np.array(right_seed, dtype=np.float32)
		logger.info("Converting %d EE steps to joint commands via IK...", chunk_size)
		for t in range(chunk_size):
			left_slice = ee_actions[t, : self._arm_action_dim]
			right_slice = ee_actions[t, self._arm_action_dim :]
			left_joints, left_gripper = self._solve_arm_ik(
				left_slice, left_seed, last_left, arm_label="left"
			)
			right_joints, right_gripper = self._solve_arm_ik(
				right_slice, right_seed, last_right, arm_label="right"
			)

			joint_actions[t, :6] = left_joints
			joint_actions[t, 6] = left_gripper
			joint_actions[t, 7:13] = right_joints
			joint_actions[t, 13] = right_gripper

		self._ik_seed_left = last_left.astype(np.float32).tolist()
		self._ik_seed_right = last_right.astype(np.float32).tolist()

		return joint_actions

	def _solve_arm_ik(
		self,
		arm_chunk: np.ndarray,
		seed: List[float],
		last_valid: np.ndarray,
		arm_label: str,
	) -> tuple[np.ndarray, float]:
		pos = arm_chunk[:3]
		orientation = arm_chunk[3 : 3 + self._action_orientation_dim]
		gripper = float(arm_chunk[3 + self._action_orientation_dim])
		roll, pitch, yaw = orientation.tolist()
		pose = np.array([*pos, roll, pitch, yaw], dtype=np.float32)
		try:
			joints = np.array(
				calculate_inverse_kinematics(pose.tolist(), seed=seed), dtype=np.float32
			)
			last_valid[:] = joints
			seed[:] = joints.tolist()
		except Exception as exc:  # pragma: no cover - IK failures are rare in tests
			logger.warning("IK failed for %s arm: %s. Reusing last valid solution.", arm_label, exc)
			joints = last_valid
		return joints, gripper


def _ensure_flag(policy_args: List[str], prefix: str, value: str) -> None:
	if any(arg.startswith(prefix) for arg in policy_args):
		return
	policy_args.append(f"{prefix}{value}")


def parse_runtime_config() -> MsraPi05RuntimeConfig:
	parser = argparse.ArgumentParser(
		"MSRA Pi05 inference server",
		description=(
			"Runs a socket server that receives joint observations, converts them to end-effector"
			" inputs, executes a Pi05 policy, and replies with joint action chunks.\n"
			"Unknown CLI arguments are forwarded to EvalPipelineConfig (e.g. --policy.path, --env.type)."
		),
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)
	parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
	parser.add_argument("--host", type=str, default=ServerConfig.host)
	parser.add_argument("--port", type=int, default=ServerConfig.port)
	parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
	parser.add_argument("--chunk-size", type=int, default=50)
	parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float32", "float16"])
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--resize-width", type=int, default=224)
	parser.add_argument("--resize-height", type=int, default=224)
	parser.add_argument("--add-new-tokens", dest="add_new_tokens", action="store_true")
	parser.add_argument("--no-add-new-tokens", dest="add_new_tokens", action="store_false")
	parser.set_defaults(add_new_tokens=True)
	parser.add_argument("--language-fallback", type=str, default="")
	parser.add_argument("--policy-type", type=str, default="pi05")
	parser.add_argument("--env-type", type=str, default="msra-ee")

	args, remaining = parser.parse_known_args()
	policy_args = list(remaining)
	if not policy_args:
		policy_args.extend([
			f"--policy.type={args.policy_type}",
			f"--env.type={args.env_type}",
		])
	else:
		_ensure_flag(policy_args, "--policy.type=", args.policy_type)
		_ensure_flag(policy_args, "--env.type=", args.env_type)

	if args.device and not any(arg.startswith("--device=") for arg in policy_args):
		policy_args.append(f"--device={args.device}")
	eval_cfg = draccus.parse(EvalPipelineConfig, args=policy_args)
	if args.device:
		logger.info("Using device override: %s", args.device)

	return MsraPi05RuntimeConfig(
		host=args.host,
		port=args.port,
		device=args.device,
		checkpoint_path=args.checkpoint,
		chunk_size=args.chunk_size,
		dtype=args.dtype,
		seed=args.seed,
		resize_width=args.resize_width,
		resize_height=args.resize_height,
		add_new_tokens=args.add_new_tokens,
		language_fallback=args.language_fallback,
		eval_cfg=eval_cfg,
	)


def main() -> None:
	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s - %(levelname)s - %(message)s",
	)
	config = parse_runtime_config()
	server = MsraPi05AlohaServer(config)
	server.run()


if __name__ == "__main__":  # pragma: no cover
	main()
