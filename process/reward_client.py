"""Query Robometer via the same trajectory-prefix RPC used by online relabeling."""

import argparse
from io import BytesIO
from pathlib import Path

import grpc
import h5py
import numpy as np

from robometer_policy_learning.distributed.protos import learner_pb2, reward_relabel_pb2, reward_relabel_pb2_grpc


class RewardClient:
    def __init__(self, address, timeout=60, max_msg_mb=256, context_frames=None):
        limit = max_msg_mb * 1024 * 1024
        self.channel = grpc.insecure_channel(address, options=[
            ("grpc.enable_http_proxy", 0), ("grpc.max_send_message_length", limit),
            ("grpc.max_receive_message_length", limit),
        ])
        self.stub = reward_relabel_pb2_grpc.RewardRelabelServiceStub(self.channel)
        self.timeout = timeout
        self.context_frames = context_frames

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.channel.close()

    def predict(self, frames, indices, task, episode_id, camera="cam_high"):
        indices = [int(index) for index in indices]
        if not indices or any(index < 0 or index >= len(frames) for index in indices) or not task:
            raise ValueError("A task and valid frame indices are required")
        if self.context_frames is not None:
            if self.context_frames < 1:
                raise ValueError("Reward context length must be positive")
            # Match Robometer's prefix subsampling exactly, without sending an
            # ever-growing full RGB episode through a bounded gRPC message.
            results = []
            for index in indices:
                if index + 1 <= self.context_frames:
                    selected = range(index + 1)
                elif self.context_frames == 1:
                    selected = [index]
                else:
                    selected = np.rint(np.linspace(0, index, self.context_frames)).astype(int)
                clip = [frames[int(i)] for i in selected]
                results.append(self._predict(clip, [len(clip) - 1], task, episode_id, camera))
            return np.concatenate(results, axis=1)
        return self._predict(frames, indices, task, episode_id, camera)

    def _predict(self, frames, indices, task, episode_id, camera):
        transitions = []
        for frame in frames:
            buffer = BytesIO()
            np.save(buffer, frame, allow_pickle=False)
            transitions.append(learner_pb2.Transition(obs={camera: learner_pb2.NDArray(data=buffer.getvalue())}))
        reply = self.stub.RelabelRewards(reward_relabel_pb2.RelabelRewardsRequest(
            transitions=transitions, language_instructions=[task], episode_ids=[episode_id], batch_indices=indices,
        ), timeout=self.timeout)
        if not reply.ok or camera not in reply.progress_predictions_by_key:
            raise RuntimeError(f"Robometer did not score {episode_id}: {reply.message}")
        result = reply.progress_predictions_by_key[camera]
        values = np.asarray([result.progress, result.success_probs], dtype=np.float32)
        if values.shape != (2, len(indices)) or not np.isfinite(values).all() or not ((0 <= values) & (values <= 1)).all():
            raise ValueError(f"Invalid reward response for {episode_id}")
        return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", default="127.0.0.1:50052")
    parser.add_argument("--replay", required=True, type=Path)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--frames", required=True, type=int)
    args = parser.parse_args()
    with h5py.File(args.replay, "r") as data, RewardClient(args.address) as client:
        episode = data["data"][args.episode]
        frames = episode["obs/cam_high"][:args.frames]
        task = episode["language_instruction"].asstr()[()]
        progress, success = client.predict(frames, [len(frames) - 1], task, args.episode)
        print(f"progress={progress[0]:.6f} success={success[0]:.6f}")


if __name__ == "__main__":
    main()
