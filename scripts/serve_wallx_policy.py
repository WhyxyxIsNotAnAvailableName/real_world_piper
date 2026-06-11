"""Serve the WallX pi0.5 policy behind a robot-friendly websocket API.

Example:
    uv run --active scripts/serve_wallx_policy.py \
        --port 8000

Client request schema:
    {
        "instruction": str,
        "base_image": uint8 ndarray [H, W, 3],
        "left_wrist_image": uint8 ndarray [H, W, 3],
        "right_wrist_image": uint8 ndarray [H, W, 3],
        "joint_degrees": float32 ndarray [6],
        "gripper": float,  # 0=open, 1=closed
    }

Response schema:
    {
        "actions": float32 ndarray [10, 7],
        # actions[:, :6] are absolute joint targets in degrees.
        # actions[:, 6] is binary gripper command: 0=open, 1=closed.
    }
"""

from __future__ import annotations

import asyncio
import dataclasses
import http
import logging
import time
import traceback
from typing import Any

import numpy as np
from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import tyro
import websockets
import websockets.asyncio.server as _server
import websockets.frames

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


WALLX_INSTRUCTIONS = (
    "Put lemon on the beige plate",
    "Put banana on the blue bowl",
    "Put donut on the pink bowl",
    "Put avocado on the purple plate",
)


@dataclasses.dataclass
class Args:
    """Arguments for the WallX policy server."""

    # Training config used to build the policy.
    config: str = "pi05_wallx"

    # Checkpoint directory, e.g. checkpoints/pi05_wallx/<exp>/19999.
    checkpoint_dir: str = (
        "/diff/wallx_workspace/openpi/checkpoints/pi05_wallx/"
        "wallx_960x540_delta_ah10_bs16_lr5e-5_20k/19999"
    )

    # Host to bind. Use 127.0.0.1 with SSH port forwarding.
    host: str = "127.0.0.1"

    # Port to serve the websocket policy on.
    port: int = 8000

    # Number of denoising steps used by pi0/pi0.5 sample_actions.
    sample_steps: int = 10

    # Optional fallback instruction if the request does not include one.
    default_instruction: str | None = None

    # Threshold for converting gripper model output to binary command.
    gripper_threshold: float = 0.5

    # Record requests and responses for debugging.
    record: bool = False

    # Directory used when --record is enabled.
    record_dir: str = "policy_records/wallx"


class WallXPolicyAdapter(_base_policy.BasePolicy):
    """Adapts a simple WallX robot request into the OpenPI policy format."""

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        *,
        default_instruction: str | None,
        gripper_threshold: float,
    ) -> None:
        self._policy = policy
        self._default_instruction = default_instruction
        self._gripper_threshold = gripper_threshold

    def infer(self, obs: dict) -> dict:
        policy_obs = self._to_policy_obs(obs)
        result = self._policy.infer(policy_obs)

        actions_radians = np.asarray(result["actions"], dtype=np.float32)
        actions = actions_radians.copy()
        actions[..., :6] = np.rad2deg(actions[..., :6])
        actions[..., 6] = (actions[..., 6] > self._gripper_threshold).astype(np.float32)

        response = dict(result)
        response["actions"] = actions
        return response

    def _to_policy_obs(self, obs: dict) -> dict:
        instruction = obs.get("instruction", obs.get("prompt", self._default_instruction))
        if instruction is None:
            raise ValueError(
                "Request must include an 'instruction' string. Valid WallX instructions are: "
                + "; ".join(WALLX_INSTRUCTIONS)
            )
        if not isinstance(instruction, str):
            raise TypeError(f"'instruction' must be a string, got {type(instruction).__name__}")

        joints = _as_vector(obs, "joint_degrees", expected_size=6)
        gripper = _as_scalar(obs, "gripper")
        gripper = 1.0 if gripper > self._gripper_threshold else 0.0
        state = np.concatenate([np.deg2rad(joints), np.array([gripper], dtype=np.float32)]).astype(np.float32)

        return {
            "observation/base_image": _as_image(obs, "base_image"),
            "observation/left_wrist_image": _as_image(obs, "left_wrist_image"),
            "observation/right_wrist_image": _as_image(obs, "right_wrist_image"),
            "observation/state": state,
            "prompt": instruction,
        }


class WallXWebsocketPolicyServer:
    """Websocket policy server with HTTP and websocket help endpoints."""

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        *,
        host: str,
        port: int,
        metadata: dict[str, Any],
        help_payload: dict[str, Any],
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata
        self._help_payload = help_payload
        self._help_text = _format_help_text(help_payload)
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=self._process_request,
        ) as server:
            await server.serve_forever()

    def _process_request(
        self, connection: _server.ServerConnection, request: _server.Request
    ) -> _server.Response | None:
        if request.path == "/healthz":
            return connection.respond(http.HTTPStatus.OK, "OK\n")
        if request.path == "/help":
            return connection.respond(http.HTTPStatus.OK, self._help_text)
        return None

    async def _handler(self, websocket: _server.ServerConnection) -> None:
        logging.info("Connection from %s opened", websocket.remote_address)
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                raw_message = await websocket.recv()
                request = _decode_request(raw_message)

                if _is_help_request(request):
                    response = {"help": self._help_payload}
                else:
                    infer_time = time.monotonic()
                    response = self._policy.infer(request)
                    infer_time = time.monotonic() - infer_time
                    response["server_timing"] = {
                        "infer_ms": infer_time * 1000,
                    }
                    if prev_total_time is not None:
                        response["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(response))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logging.info("Connection from %s closed", websocket.remote_address)
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _decode_request(raw_message: bytes | str) -> Any:
    if isinstance(raw_message, str):
        return raw_message
    return _decode_msgpack_numpy_maps(msgpack_numpy.unpackb(raw_message))


def _decode_msgpack_numpy_maps(value: Any) -> Any:
    """Decode ndarray maps produced by Python and non-Python msgpack clients.

    OpenPI's msgpack_numpy helper emits byte-string keys such as b"__ndarray__".
    Some clients naturally emit string keys such as "__ndarray__" instead. The
    shared OpenPI decoder only handles the byte-key form, so this server accepts
    both on the wire.
    """

    if isinstance(value, dict):
        if _map_get(value, "__ndarray__", default=False):
            data = _map_get(value, "data")
            dtype = np.dtype(_map_get(value, "dtype"))
            shape = tuple(_map_get(value, "shape"))
            return np.frombuffer(data, dtype=dtype).reshape(shape)

        if _map_get(value, "__npgeneric__", default=False):
            dtype = np.dtype(_map_get(value, "dtype"))
            return dtype.type(_map_get(value, "data"))

        return {
            _decode_msgpack_numpy_maps(k): _decode_msgpack_numpy_maps(v)
            for k, v in value.items()
        }

    if isinstance(value, list):
        return [_decode_msgpack_numpy_maps(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_decode_msgpack_numpy_maps(v) for v in value)
    return value


def _map_get(mapping: dict, key: str, *, default: Any = None) -> Any:
    if key in mapping:
        return mapping[key]
    return mapping.get(key.encode("utf-8"), default)


def _is_help_request(request: Any) -> bool:
    if isinstance(request, bytes):
        try:
            request = request.decode("utf-8")
        except UnicodeDecodeError:
            return False
    if isinstance(request, str):
        return request.strip().lower() in {"help", "/help"}
    if isinstance(request, dict):
        return bool(request.get("help", False)) or str(request.get("command", "")).strip().lower() == "help"
    return False


def _as_vector(obs: dict, key: str, *, expected_size: int) -> np.ndarray:
    if key not in obs:
        raise KeyError(f"Request is missing required field '{key}'")
    value = np.asarray(obs[key], dtype=np.float32).reshape(-1)
    if value.shape != (expected_size,):
        raise ValueError(f"'{key}' must have shape ({expected_size},), got {value.shape}")
    return value


def _as_scalar(obs: dict, key: str) -> float:
    if key not in obs:
        raise KeyError(f"Request is missing required field '{key}'")
    value = np.asarray(obs[key], dtype=np.float32)
    if value.size != 1:
        raise ValueError(f"'{key}' must be a scalar, got shape {value.shape}")
    return float(value.reshape(()))


def _as_image(obs: dict, key: str) -> np.ndarray:
    if key not in obs:
        raise KeyError(f"Request is missing required field '{key}'")

    image = np.asarray(obs[key])
    if image.ndim != 3:
        raise ValueError(f"'{key}' must be an HWC or CHW image, got shape {image.shape}")
    if image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] == 4:
        image = image[..., :3]
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    if image.shape[-1] != 3:
        raise ValueError(f"'{key}' must have 3 RGB channels, got shape {image.shape}")

    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating) and np.nanmax(image) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def _metadata(args: Args, base_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    metadata = dict(base_metadata or {})
    metadata.update(
        {
            "robot": "wallx_right_arm",
            "config": args.config,
            "checkpoint_dir": args.checkpoint_dir,
            "request_schema": {
                "instruction": "str",
                "base_image": "uint8 ndarray [H, W, 3], RGB",
                "left_wrist_image": "uint8 ndarray [H, W, 3], RGB",
                "right_wrist_image": "uint8 ndarray [H, W, 3], RGB",
                "joint_degrees": "float ndarray [6], absolute current joints in degrees",
                "gripper": "float scalar, 0=open and 1=closed",
            },
            "response_schema": {
                "actions": "float32 ndarray [10, 7]; first 6 dims absolute joint targets in degrees, "
                "last dim binary gripper 0=open/1=closed",
            },
            "valid_instructions": list(WALLX_INSTRUCTIONS),
            "action_horizon": 10,
            "dataset_fps": 5,
            "sample_steps": args.sample_steps,
            "gripper_threshold": args.gripper_threshold,
            "help": {
                "http": f"http://{args.host}:{args.port}/help",
                "websocket": '"help" or {"help": true}',
            },
            "wire_format": {
                "transport": "websocket binary frames",
                "encoding": "msgpack with OpenPI msgpack_numpy ndarray extension",
                "python_client": "openpi_client.websocket_client_policy.WebsocketClientPolicy",
                "ndarray_encoding": {
                    "__ndarray__": True,
                    "data": "raw ndarray.tobytes()",
                    "dtype": "numpy dtype string, e.g. '|u1' or '<f4'",
                    "shape": "array shape tuple/list",
                },
                "notes": [
                    "The server sends one msgpack metadata frame immediately after websocket connection.",
                    "Each inference request is one msgpack dict frame.",
                    "Each inference response is one msgpack dict frame.",
                    "The server accepts ndarray maps with either string keys or byte-string keys.",
                    "HTTP GET /help is plain text; websocket help response is msgpack.",
                ],
            },
        }
    )
    return metadata


def _help_payload(args: Args, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "summary": "WallX pi0.5 policy server",
        "server": {
            "host": args.host,
            "port": args.port,
            "config": args.config,
            "checkpoint_dir": args.checkpoint_dir,
            "sample_steps": args.sample_steps,
        },
        "request_schema": metadata["request_schema"],
        "response_schema": metadata["response_schema"],
        "wire_format": metadata["wire_format"],
        "valid_instructions": metadata["valid_instructions"],
        "units": {
            "request_joint_degrees": "absolute current joint angles in degrees",
            "internal_joint_radians": "server converts request joints to radians before policy inference",
            "response_joint_degrees": "absolute target joint angles in degrees",
            "gripper": "0=open, 1=closed",
        },
        "ssh_forwarding": {
            "command": "ssh -N -L 8000:127.0.0.1:8000 user@gpu-server",
            "client_endpoint_after_forwarding": "127.0.0.1:8000",
        },
        "health_check": f"http://{args.host}:{args.port}/healthz",
    }


def _format_help_text(help_payload: dict[str, Any]) -> str:
    lines = [
        "WallX pi0.5 policy server",
        "",
        "HTTP:",
        "  GET /healthz -> OK",
        "  GET /help    -> this help text",
        "",
        "Websocket help:",
        '  send "help" or {"help": true}',
        "",
        "Wire format:",
        "  Transport: websocket binary frames",
        "  Encoding: msgpack with OpenPI msgpack_numpy ndarray extension",
        "  Python client: openpi_client.websocket_client_policy.WebsocketClientPolicy",
        "  Connection: server sends one msgpack metadata frame first",
        "  Request: one msgpack dict frame per inference",
        "  Response: one msgpack dict frame per inference",
        "  ndarray map:",
        '    {"__ndarray__": true, "data": raw bytes, "dtype": numpy dtype str, "shape": shape}',
        "  ndarray map keys may be strings or byte strings",
        "",
        "Request schema:",
    ]
    for key, value in help_payload["request_schema"].items():
        lines.append(f"  {key}: {value}")

    lines.extend(["", "Response schema:"])
    for key, value in help_payload["response_schema"].items():
        lines.append(f"  {key}: {value}")

    lines.extend(["", "Valid instructions:"])
    for instruction in help_payload["valid_instructions"]:
        lines.append(f"  - {instruction}")

    lines.extend(
        [
            "",
            "Units:",
            "  request joint_degrees: absolute current joint angles in degrees",
            "  response actions[:, :6]: absolute target joint angles in degrees",
            "  gripper: 0=open, 1=closed",
            "",
            "SSH forwarding:",
            f"  {help_payload['ssh_forwarding']['command']}",
            "",
        ]
    )
    return "\n".join(lines)


def main(args: Args) -> None:
    base_policy = _policy_config.create_trained_policy(
        _config.get_config(args.config),
        args.checkpoint_dir,
        sample_kwargs={"num_steps": args.sample_steps},
    )
    policy: _base_policy.BasePolicy = WallXPolicyAdapter(
        base_policy,
        default_instruction=args.default_instruction,
        gripper_threshold=args.gripper_threshold,
    )
    if args.record:
        policy = _policy.PolicyRecorder(policy, args.record_dir)

    logging.info("Serving WallX policy on %s:%d", args.host, args.port)
    logging.info("Checkpoint: %s", args.checkpoint_dir)
    logging.info("Valid instructions: %s", "; ".join(WALLX_INSTRUCTIONS))

    metadata = _metadata(args, getattr(base_policy, "metadata", None))
    help_payload = _help_payload(args, metadata)

    server = WallXWebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=metadata,
        help_payload=help_payload,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
