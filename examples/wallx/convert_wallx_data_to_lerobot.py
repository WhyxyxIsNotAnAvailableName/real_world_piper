"""
Convert WallX real-world HDF5 episodes to a LeRobot dataset.

Example:
    uv run examples/wallx/convert_wallx_data_to_lerobot.py \
        --raw-dir /diff/wallx_workspace/wallx_data_ckp/real_world \
        --repo-id wallx/real_world

The output dataset is written under the LeRobot cache directory for the given
repo id. By default, joint angles are converted from degrees to radians and the
gripper is binarized per episode.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import shutil
from typing import Literal

import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from PIL import Image
from tqdm import tqdm
import tyro


TASKS = {
    "lemon": "Put lemon on the beige plate",
    "banana": "Put banana on the blue bowl",
    "donut": "Put donut on the pink bowl",
    "avocado": "Put avocado on the purple plate",
}

CAMERAS = {
    "base_image": "camera_l515",
    "left_wrist_image": "camera_f",
    "right_wrist_image": "camera_r",
}


@dataclasses.dataclass(frozen=True)
class ThresholdInfo:
    threshold: float
    lower_value: float
    upper_value: float
    gap: float
    min_value: float
    max_value: float
    num_values: int
    num_unique: int
    method: str


def infer_binary_threshold(values: np.ndarray, *, min_cluster_fraction: float = 0.02) -> ThresholdInfo:
    """Infer a binary split threshold from one episode of gripper values.

    The threshold is placed at the midpoint of the largest gap between sorted
    unique values, while requiring at least a small number of samples on each
    side to avoid choosing a single noisy outlier when episodes are long.
    """

    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        raise ValueError("Cannot infer gripper threshold from empty or non-finite values.")

    unique, counts = np.unique(flat, return_counts=True)
    if unique.size == 1:
        value = float(unique[0])
        return ThresholdInfo(
            threshold=value,
            lower_value=value,
            upper_value=value,
            gap=0.0,
            min_value=value,
            max_value=value,
            num_values=int(flat.size),
            num_unique=1,
            method="constant",
        )

    min_side_count = max(1, min(10, int(round(flat.size * min_cluster_fraction))))
    cumulative = np.cumsum(counts)
    left_counts = cumulative[:-1]
    right_counts = flat.size - cumulative[:-1]
    eligible = (left_counts >= min_side_count) & (right_counts >= min_side_count)

    gaps = unique[1:] - unique[:-1]
    if np.any(eligible):
        eligible_indices = np.flatnonzero(eligible)
        best_index = int(eligible_indices[np.argmax(gaps[eligible])])
        method = "largest_gap"
    else:
        best_index = int(np.argmax(gaps))
        method = "largest_gap_no_min_cluster"

    lower = float(unique[best_index])
    upper = float(unique[best_index + 1])
    return ThresholdInfo(
        threshold=(lower + upper) / 2.0,
        lower_value=lower,
        upper_value=upper,
        gap=float(upper - lower),
        min_value=float(unique[0]),
        max_value=float(unique[-1]),
        num_values=int(flat.size),
        num_unique=int(unique.size),
        method=method,
    )


def resize_image(image: np.ndarray, *, width: int, height: int, mode: Literal["pad", "stretch"]) -> np.ndarray:
    image = np.asarray(image)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.shape[:2] == (height, width):
        return image

    pil_image = Image.fromarray(image)
    if mode == "stretch":
        return np.asarray(pil_image.resize((width, height), resample=Image.BICUBIC))
    if mode != "pad":
        raise ValueError(f"Unsupported resize mode: {mode}")

    src_width, src_height = pil_image.size
    scale = min(width / src_width, height / src_height)
    resized_width = max(1, round(src_width * scale))
    resized_height = max(1, round(src_height * scale))
    resized = pil_image.resize((resized_width, resized_height), resample=Image.BICUBIC)

    canvas = Image.new("RGB", (width, height))
    left = (width - resized_width) // 2
    top = (height - resized_height) // 2
    canvas.paste(resized, (left, top))
    return np.asarray(canvas)


def convert_robot_values(
    values: np.ndarray,
    *,
    gripper_threshold: float,
    gripper_high_value: Literal["open", "closed"],
) -> np.ndarray:
    converted = np.asarray(values, dtype=np.float32).copy()
    converted[..., :6] = np.deg2rad(converted[..., :6])

    raw_gripper = np.asarray(values[..., 6], dtype=np.float32)
    if gripper_high_value == "open":
        # openpi convention: 0.0 is open, 1.0 is closed.
        gripper = raw_gripper < gripper_threshold
    elif gripper_high_value == "closed":
        gripper = raw_gripper >= gripper_threshold
    else:
        raise ValueError(f"Unsupported gripper_high_value: {gripper_high_value}")

    converted[..., 6] = gripper.astype(np.float32)
    return converted


def get_episode_paths(raw_dir: Path) -> list[tuple[str, Path]]:
    episodes: list[tuple[str, Path]] = []
    for object_name in TASKS:
        object_dir = raw_dir / object_name
        if not object_dir.exists():
            raise FileNotFoundError(f"Expected object directory not found: {object_dir}")
        for episode_path in sorted(object_dir.glob("episode_*.hdf5")):
            episodes.append((object_name, episode_path))
    return episodes


def validate_episode(ep: h5py.File, episode_path: Path) -> None:
    required_paths = [
        "action",
        "observations/qpos",
        *(f"observations/images/{camera}" for camera in CAMERAS.values()),
    ]
    for key in required_paths:
        if key not in ep:
            raise KeyError(f"{episode_path} is missing required dataset: {key}")


def main(
    raw_dir: Path = Path("/diff/wallx_workspace/wallx_data_ckp/real_world"),
    repo_id: str = "wallx/real_world",
    *,
    fps: int = 5,
    image_width: int = 320,
    image_height: int = 180,
    resize_mode: Literal["pad", "stretch"] = "pad",
    robot_type: str = "wallx_right_arm",
    overwrite: bool = True,
    use_videos: bool = True,
    image_writer_processes: int = 5,
    image_writer_threads: int = 10,
    gripper_high_value: Literal["open", "closed"] = "open",
    shared_gripper_threshold: bool = False,
    min_gripper_cluster_fraction: float = 0.02,
    max_episodes: int | None = None,
    push_to_hub: bool = False,
) -> None:
    raw_dir = raw_dir.expanduser().resolve()
    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"Output dataset already exists: {output_path}")
        shutil.rmtree(output_path)

    features = {
        "state": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["state"],
        },
        "actions": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["actions"],
        },
    }
    for feature_name in CAMERAS:
        features[feature_name] = {
            "dtype": "video" if use_videos else "image",
            "shape": (image_height, image_width, 3),
            "names": ["height", "width", "channel"],
        }

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type=robot_type,
        fps=fps,
        features=features,
        use_videos=use_videos,
        image_writer_processes=image_writer_processes,
        image_writer_threads=image_writer_threads,
    )

    report_rows = []
    total_frames = 0
    all_episodes = get_episode_paths(raw_dir)
    episodes = all_episodes
    if max_episodes is not None:
        episodes = all_episodes[:max_episodes]
    print(f"Found {len(all_episodes)} episodes under {raw_dir}; converting {len(episodes)}.")

    for object_name, episode_path in tqdm(episodes, desc="Converting episodes"):
        task = TASKS[object_name]

        with h5py.File(episode_path, "r") as ep:
            validate_episode(ep, episode_path)

            raw_actions = np.asarray(ep["action"][:], dtype=np.float32)
            raw_state = np.asarray(ep["observations/qpos"][:], dtype=np.float32)
            if raw_actions.shape != raw_state.shape:
                raise ValueError(
                    f"{episode_path} has mismatched action/state shapes: "
                    f"{raw_actions.shape} vs {raw_state.shape}"
                )
            if raw_actions.ndim != 2 or raw_actions.shape[1] != 7:
                raise ValueError(f"{episode_path} expected action/state shape (T, 7), got {raw_actions.shape}")

            if shared_gripper_threshold:
                combined_threshold = infer_binary_threshold(
                    np.concatenate([raw_state[:, 6], raw_actions[:, 6]]),
                    min_cluster_fraction=min_gripper_cluster_fraction,
                )
                state_threshold = combined_threshold
                action_threshold = combined_threshold
            else:
                state_threshold = infer_binary_threshold(
                    raw_state[:, 6],
                    min_cluster_fraction=min_gripper_cluster_fraction,
                )
                action_threshold = infer_binary_threshold(
                    raw_actions[:, 6],
                    min_cluster_fraction=min_gripper_cluster_fraction,
                )

            state = convert_robot_values(
                raw_state,
                gripper_threshold=state_threshold.threshold,
                gripper_high_value=gripper_high_value,
            )
            actions = convert_robot_values(
                raw_actions,
                gripper_threshold=action_threshold.threshold,
                gripper_high_value=gripper_high_value,
            )

            image_datasets = {
                feature_name: ep[f"observations/images/{camera_name}"] for feature_name, camera_name in CAMERAS.items()
            }
            num_frames = raw_state.shape[0]
            for i in range(num_frames):
                frame = {
                    "state": state[i],
                    "actions": actions[i],
                    "task": task,
                }
                for feature_name, image_dataset in image_datasets.items():
                    frame[feature_name] = resize_image(
                        image_dataset[i],
                        width=image_width,
                        height=image_height,
                        mode=resize_mode,
                    )
                dataset.add_frame(frame)

            dataset.save_episode()

        total_frames += int(num_frames)
        report_rows.append(
            {
                "object": object_name,
                "episode": str(episode_path),
                "task": task,
                "num_frames": int(num_frames),
                "state_gripper_threshold": dataclasses.asdict(state_threshold),
                "action_gripper_threshold": dataclasses.asdict(action_threshold),
            }
        )

    report_path = output_path / "wallx_conversion_report.jsonl"
    with report_path.open("w", encoding="utf-8") as f:
        for row in report_rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    if push_to_hub:
        dataset.push_to_hub(
            tags=["wallx", "real-world"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )

    print(f"Converted {len(episodes)} episodes and {total_frames} frames.")
    print(f"LeRobot dataset: {output_path}")
    print(f"Threshold report: {report_path}")


if __name__ == "__main__":
    tyro.cli(main)
