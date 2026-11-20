"""Convert the registered DenseRL C-only split to Robometer HDF5 replay input."""

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import decord
import h5py
import numpy as np
import pyarrow.parquet as pq


SOURCES = {"c-only-offline": "offline", "c-only-online": "online"}
CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
STATE_COLUMN = "observations.state.qpos"
ACTION_COLUMN = "action.qpos"


def write_observation(obs, next_obs, key, frames):
    source = h5py.VirtualSource(".", frames.name, shape=frames.shape)
    for group, rows in ((obs, slice(None, -1)), (next_obs, slice(1, None))):
        layout = h5py.VirtualLayout((len(frames) - 1, *frames.shape[1:]), dtype=frames.dtype)
        layout[:] = source[rows]
        group.create_virtual_dataset(key, layout)


def write_episode(group, root, info, episode, task, height, width, max_transitions=None):
    index = episode["episode_index"]
    path_args = {"episode_chunk": index // info["chunks_size"], "episode_index": index}
    table = pq.read_table(root / info["data_path"].format(**path_args), columns=[
        STATE_COLUMN, ACTION_COLUMN, "timestamp", "frame_index",
    ])
    frame_count = len(table)
    if frame_count < 2 or frame_count != episode["frames"]:
        raise ValueError(f"Frame count differs from registered episode {episode['id']}")
    if not np.array_equal(table["frame_index"].to_numpy(), np.arange(frame_count)):
        raise ValueError(f"Non-contiguous frame indices for {episode['id']}")
    states = np.asarray(table[STATE_COLUMN].to_pylist(), dtype=np.float32)
    actions = np.asarray(table[ACTION_COLUMN].to_pylist(), dtype=np.float32)
    timestamps = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64).reshape(-1)
    if (states.shape != (frame_count, 14) or actions.shape != states.shape
            or not all(np.isfinite(values).all() for values in (states, actions, timestamps))
            or not (np.diff(timestamps) > 0).all()):
        raise ValueError(f"Invalid state, action or timestamps for {episode['id']}")

    recorded_frames = frame_count
    if max_transitions is not None:
        frame_count = min(frame_count, max_transitions + 1)
        states, actions, timestamps = states[:frame_count], actions[:frame_count], timestamps[:frame_count]
    group.attrs.update({"source": episode["source"], "episode_index": index, "outcome": episode["outcome"]})
    group.create_dataset("actions", data=actions[:-1])
    dones = np.zeros(frame_count - 1, dtype=np.bool_)
    dones[-1] = frame_count == recorded_frames
    group.create_dataset("dones", data=dones)
    group.create_dataset("timestamps", data=timestamps)
    vr_ranges = [(start, min(end, frame_count - 1)) for start, end in episode["vr_ranges"] if start < frame_count]
    group.create_dataset("vr_ranges", data=np.asarray(vr_ranges, dtype=np.int64).reshape(-1, 2))
    group.create_dataset("language_instruction", data=task, dtype=h5py.string_dtype("utf-8"))

    frames = group.create_group("frames")
    obs, next_obs = group.create_group("obs"), group.create_group("next_obs")
    state = frames.create_dataset("state", data=states)
    write_observation(obs, next_obs, "state", state)
    for camera in CAMERAS:
        path = root / info["video_path"].format(video_key=f"observation.images.{camera}", **path_args)
        reader = decord.VideoReader(str(path), width=width, height=height, num_threads=1)
        if len(reader) != recorded_frames:
            raise ValueError(f"Video and Parquet frame counts differ: {path}")
        images = frames.create_dataset(camera, shape=(frame_count, height, width, 3), dtype=np.uint8,
                                       chunks=(1, height, width, 3), compression="lzf")
        for start in range(0, frame_count, 32):
            end = min(start + 32, frame_count)
            images[start:end] = reader.get_batch(list(range(start, end))).asnumpy()
        write_observation(obs, next_obs, camera, images)
    return frame_count - 1


def prepare(config_path, data_root, split, output, height, width, episode_id, max_transitions=None):
    if min(height, width) < 1:
        raise ValueError("Image dimensions must be positive")
    if max_transitions is not None and (episode_id is None or max_transitions < 1):
        raise ValueError("A positive transition limit requires an explicit smoke-test episode ID")
    config = json.loads(config_path.read_text())
    manifest_path = config_path.parent / config["manifest"]
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    episodes = [row for row in manifest["splits"][split] if episode_id is None or row["id"] == episode_id]
    if not episodes or len({row["id"] for row in episodes}) != len(episodes):
        raise ValueError(f"Expected a nonempty list of unique {split} episodes")
    sources = {}
    for name in {row["source"] for row in episodes}:
        root = data_root / SOURCES[name]
        metadata = root / "meta/episodes.jsonl"
        if hashlib.sha256(metadata.read_bytes()).hexdigest() != manifest["sources"][name]["sha256"]:
            raise ValueError(f"Dataset metadata differs from the registered snapshot: {metadata}")
        sources[name] = root, json.loads((root / "meta/info.json").read_text())

    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".iql-", dir=output.parent) as temporary:
        pending = Path(temporary) / "replay.h5"
        with h5py.File(pending, "w", libver="latest") as file:
            file.attrs.update({
                "split": split,
                "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                "robot_type": "r1_lite",
                "action_column": ACTION_COLUMN,
                "state_column": STATE_COLUMN,
                "complete_split": episode_id is None,
                "reward_status": "pending_robometer_relabel",
            })
            groups = file.create_group("data", track_order=True)
            for episode in episodes:
                root, info = sources[episode["source"]]
                steps = write_episode(groups.create_group(episode["id"]), root, info, episode,
                                      config["task"], height, width, max_transitions)
                print(f"{split} {episode['id']} {steps} transitions", flush=True)
        pending.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-config", type=Path, required=True, help="Robometer prepare.json and its episode split")
    parser.add_argument("--data-root", type=Path, required=True, help="Directory containing offline/ and online/")
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--episode-id", help="Convert one UUID for a smoke check")
    parser.add_argument("--max-transitions", type=int, help="Limit the selected smoke-test episode to this prefix")
    args = parser.parse_args()
    prepare(args.prepare_config, args.data_root, args.split, args.output, args.height, args.width,
            args.episode_id, args.max_transitions)
