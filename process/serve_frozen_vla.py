"""Serve frozen R1 Lite OpenPI features and explicitly supplied diffusion noise."""

import argparse
from pathlib import Path

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from openpi.models import model as model_api
from openpi.models.pi0 import make_attn_mask
from openpi.policies.policy_config import create_trained_policy
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.shared.nnx_utils import module_jit
from openpi.training.config import get_config
from openpi_client.base_policy import BasePolicy


class PooledVLM(nnx.Module):
    def __init__(self, model):
        self.model = model

    def encode(self, observation):
        observation = model_api.preprocess_observation(None, observation, train=False)
        tokens, mask, autoregressive = self.model.embed_prefix(observation)
        hidden, _ = self.model.PaliGemma.llm(
            [tokens, None], mask=make_attn_mask(mask, autoregressive), positions=jnp.cumsum(mask, axis=1) - 1,
        )
        return (hidden[0].astype(jnp.float32) * mask[..., None]).sum(axis=1) / mask.sum(axis=1, keepdims=True)


class NoisePolicy(BasePolicy):
    def __init__(self, policy, dino_path=None):
        if policy._is_pytorch_model:
            raise ValueError("This adapter expects the JAX OpenPI checkpoint")
        self.policy = policy
        self.horizon = policy._model.action_horizon
        self.action_dim = policy._model.action_dim
        self.encode = module_jit(PooledVLM(policy._model).encode)
        self.dino = None
        if dino_path is not None:
            from transformers import AutoImageProcessor, AutoModel

            self.dino_processor = AutoImageProcessor.from_pretrained(dino_path, local_files_only=True)
            self.dino = AutoModel.from_pretrained(dino_path, local_files_only=True).eval()

    def infer(self, observation):
        observation = dict(observation)
        operation = observation.pop("dsrl_operation")
        if operation == "features":
            inputs = self.policy._input_transform(jax.tree.map(lambda x: x, observation))
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[None], inputs)
            vlm = np.asarray(self.encode(model_api.Observation.from_dict(inputs))[0], dtype=np.float32)
            features = {"state": np.asarray(observation["state"], dtype=np.float32), "vlm_features": vlm}
            if self.dino is not None:
                import torch

                image = observation["images"]["cam_left_wrist"]
                with torch.inference_mode():
                    embedding = self.dino(**self.dino_processor(images=image, return_tensors="pt")).last_hidden_state[0, 0]
                features["dino_wrist"] = embedding.cpu().numpy()
            return {"features": features}
        if operation != "sample":
            raise ValueError(f"Unsupported DSRL operation: {operation}")
        noise = np.asarray(observation.pop("dsrl_noise"), dtype=np.float32)
        if noise.shape != (self.horizon, self.action_dim) or not np.isfinite(noise).all():
            raise ValueError(f"Expected finite noise with shape {(self.horizon, self.action_dim)}")
        return self.policy.infer(observation, noise=noise)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dino-model", type=Path)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve(strict=True)
    policy = create_trained_policy(get_config(args.config), str(checkpoint))
    adapter = NoisePolicy(policy, args.dino_model)
    WebsocketPolicyServer(adapter, host="127.0.0.1", port=args.port, metadata={
        "model_horizon": adapter.horizon, "noise_dim": adapter.action_dim,
        "vla_checkpoint": str(checkpoint), "vla_config": args.config,
        "dino_model": str(args.dino_model) if args.dino_model is not None else None,
    }).serve_forever()


if __name__ == "__main__":
    main()
