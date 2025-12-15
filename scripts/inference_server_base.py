"""Base inference server abstraction for model-driven ALOHA style policies.
This module factors out the socket communication protocol shared by several
specialized inference servers (`inference_on_aloha_server.py`,
`inference_server.py`, `inference_on_aloha_server_msra.py`).
Protocol (unchanged):
1. Client sends a single message preceded by a 4-byte big-endian unsigned int
   specifying the full message length in bytes.
2. Message body starts with a UTF-8 JSON line terminated by a single `\n`.
   The JSON contains at least: `qpos` (list[float]), `language` (str), and for
   every image key K in `image_keys` an integer field `K_size` giving byte size.
3. Immediately after the newline the concatenated raw image bytes appear in the
   order of `image_keys`. Each image is individually decodable by Pillow.
Server response:
1. Sends a 4-byte big-endian unsigned int specifying byte length of the action
   array payload.
2. Sends raw bytes of a contiguous `float32` NumPy array shaped `(chunk_size, action_dim)`.
Subclass Responsibilities:
- Implement `build_model(self)` returning the loaded model (or any object used
  for inference). Called once during initialization.
- Implement `process_observation(self, raw_obs: dict) -> dict` converting the
  raw received observation dict (images as PIL, `qpos` ndarray, `language` str)
  into a model-ready batch / structure.
- Implement `generate_actions(self, processed_obs: dict) -> np.ndarray` that
  runs model inference and returns a NumPy ndarray shaped `(chunk_size, action_dim)`.
Optional Hooks:
- `on_client_connect(self, addr)`
- `on_client_disconnect(self, addr)`
- `on_iteration(self, raw_obs, processed_obs, actions)` for per-step logging / debugging.
Threading / Concurrency:
The current design is single-connection, sequential processing to match the
original scripts. Subclasses may override `run()` to implement multi-client or
async behavior if needed without duplicating protocol code.
Graceful Shutdown:
Call `stop()` from another thread / signal handler to break the accept loop.
Example Subclass Sketch (Pi0 joint-space):
```
class Pi0JointServer(InferenceServerBase):
	def build_model(self):
		policy = PI0Policy.from_pretrained(self.checkpoint_path).to(self.device).eval()
		return policy
	def process_observation(self, raw_obs):
		# Resize images, tokenize language, normalize state, return dict
		...
	def generate_actions(self, processed):
		with torch.no_grad():
			actions = self.model.predict_action_chunk(processed)[0].cpu().numpy()
		return actions
```
"""

from __future__ import annotations

import json
import socket
import struct
import logging
import io
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence, Optional, Dict, Any

import numpy as np
from PIL import Image


def _configure_logging():
	"""Force a consistent logging configuration so base server messages appear.
	Using force=True ensures prior basicConfig calls from dependent libraries
	don't suppress INFO logs from this module.
	"""
	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s - %(levelname)s - %(message)s",
		force=True,
	)

_configure_logging()
logger = logging.getLogger(__name__)


@dataclass
class ServerConfig:
	host: str = "0.0.0.0"
	port: int = 5000
	device: str = "cuda"
	image_keys: Sequence[str] = ("cam_high", "cam_left_wrist", "cam_right_wrist")
	recv_timeout: Optional[float] = None  # seconds; None => blocking
	send_timeout: Optional[float] = None
	reuse_addr: bool = True


class InferenceServerBase(ABC):
	"""Abstract base class encapsulating network protocol and runtime loop.
	Subclasses *must* implement model loading and inference specific logic but
	benefit from shared code for:
	- Socket setup & teardown
	- Observation reception & parsing
	- Action serialization & sending
	- Iterative connection handling loop
	"""

	def __init__(self, config: ServerConfig):
		self.config = config
		self.host = config.host
		self.port = config.port
		self.device = config.device
		self.image_keys = list(config.image_keys)
		self._stop_requested = False

		# Socket setup
		self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		if config.reuse_addr:
			self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		if config.recv_timeout is not None:
			self.socket.settimeout(config.recv_timeout)

		logger.info(
			f"Initializing inference server base: host={self.host} port={self.port} device={self.device} images={self.image_keys}"
		)

		# Delegate model creation to subclass
		self.model = self.build_model()
		logger.info("Model built and ready for inference.")

	# ---------------------------------------------------------------------
	# Abstract methods to be provided by subclass
	# ---------------------------------------------------------------------
	@abstractmethod
	def build_model(self):
		"""Return the model / policy object prepared for inference."""
		raise NotImplementedError

	@abstractmethod
	def process_observation(self, raw_obs: Dict[str, Any]) -> Dict[str, Any]:
		"""Convert raw received observation dict into model-ready batch.
		Raw observation contains:
		- image keys as PIL.Image
		- 'qpos': np.ndarray (shape (14,))
		- 'language': str
		Subclasses may add normalization, resizing, tokenization, coordinate
		transforms, etc.
		"""
		raise NotImplementedError

	@abstractmethod
	def generate_actions(self, processed_obs: Dict[str, Any]) -> np.ndarray:
		"""Run model inference and return actions as float32 ndarray (chunk, dim)."""
		raise NotImplementedError

	# ---------------------------------------------------------------------
	# Optional hooks
	# ---------------------------------------------------------------------
	def on_client_connect(self, addr):  # pragma: no cover - optional
		logger.info(f"Client connected: {addr}")

	def on_client_disconnect(self, addr):  # pragma: no cover - optional
		logger.info(f"Client disconnected: {addr}")

	def on_iteration(self, raw_obs, processed_obs, actions):  # pragma: no cover
		pass

	# ---------------------------------------------------------------------
	# Network protocol helpers
	# ---------------------------------------------------------------------
	def _recv_exactly(self, conn: socket.socket, n: int) -> bytes:
		data = b""
		while len(data) < n:
			chunk = conn.recv(n - len(data))
			if not chunk:
				raise ConnectionError("Connection closed while receiving data")
			data += chunk
		return data

	def receive_observation(self, conn: socket.socket) -> Dict[str, Any]:
		"""Receive a single observation bundle using shared protocol."""
		header = self._recv_exactly(conn, 4)
		msg_size = struct.unpack("!I", header)[0]
		logger.debug(f"Expecting message size={msg_size} bytes")
		raw = self._recv_exactly(conn, msg_size)

		try:
			newline_idx = raw.index(b"\n")
		except ValueError:
			raise ValueError("Malformed message: missing JSON newline delimiter")

		meta_json = raw[:newline_idx].decode("utf-8")
		meta = json.loads(meta_json)
		offset = newline_idx + 1

		if "qpos" not in meta:
			raise ValueError("Metadata missing 'qpos'")
		if "language" not in meta:
			raise ValueError("Metadata missing 'language'")

		obs = {
			"qpos": np.array(meta["qpos"], dtype=np.float32),
			"language": meta["language"],
		}

		for key in self.image_keys:
			size_field = f"{key}_size"
			if size_field not in meta:
				raise ValueError(f"Metadata missing image size field: {size_field}")
			img_size = meta[size_field]
			img_bytes = raw[offset : offset + img_size]
			offset += img_size
			try:
				img = Image.open(io.BytesIO(img_bytes))
				img.load()  # Ensure fully read
			except Exception as e:
				raise ValueError(f"Failed to decode image '{key}': {e}") from e
			obs[key] = img

		logger.debug(
			"Received observation: "
			+ ", ".join(
				[
					f"images={len(self.image_keys)}",
					f"qpos_shape={obs['qpos'].shape}",
					f"language_len={len(obs['language'])}",
				]
			)
		)
		return obs

	def send_actions(self, conn: socket.socket, actions: np.ndarray) -> None:
		if not isinstance(actions, np.ndarray):
			raise TypeError("actions must be a numpy.ndarray")
		if actions.dtype != np.float32:
			actions = actions.astype(np.float32)
		payload = actions.tobytes()
		header = struct.pack("!I", len(payload))
		conn.sendall(header)
		conn.sendall(payload)
		logger.debug(f"Sent actions bytes={len(payload)} shape={actions.shape}")

	# ---------------------------------------------------------------------
	# Runtime control
	# ---------------------------------------------------------------------
	def stop(self):  # pragma: no cover - external control
		self._stop_requested = True
		try:
			# Trigger accept() unblock by connecting to self
			with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
				s.connect((self.host, self.port))
		except Exception:
			pass

	def run(self):  # pragma: no cover - network side effects
		"""Blocking server loop until `stop()` is called or KeyboardInterrupt."""
		self.socket.bind((self.host, self.port))
		self.socket.listen(1)
		logger.info(f"Inference server listening on {self.host}:{self.port}")
		try:
			while not self._stop_requested:
				logger.info("Waiting for client connection...")
				try:
					conn, addr = self.socket.accept()
				except socket.timeout:
					continue  # Check stop flag again
				self.on_client_connect(addr)
				try:
					while not self._stop_requested:
						logger.info("Awaiting observation message...")
						raw_obs = self.receive_observation(conn)
						processed = self.process_observation(raw_obs)
						actions = self.generate_actions(processed)
						self.send_actions(conn, actions)
						self.on_iteration(raw_obs, processed, actions)
				except ConnectionError as e:
					logger.warning(f"Connection error: {e}")
				except Exception as e:
					logger.error(f"Error during request handling: {e}", exc_info=True)
				finally:
					try:
						conn.close()
					except Exception:
						pass
					self.on_client_disconnect(addr)
		except KeyboardInterrupt:
			logger.info("Shutdown requested via KeyboardInterrupt")
		finally:
			try:
				self.socket.close()
			except Exception:
				pass
			logger.info("Socket closed; server terminated")


__all__ = ["ServerConfig", "InferenceServerBase"]
