"""Serve DenseRL's frozen H50 VLA, exact camera transforms and RTC prefix decoding.

Input: --config, the reviewed PyTorch base, normalization and DenseRL source.
Output: loopback features/cached noise decoding; never connects to the robot.
"""
import argparse
from collections import OrderedDict
import os
from pathlib import Path
import sys
import time
import uuid

import numpy as np


class NoisePolicy:
    def __init__(self, backend, dino_path, *, context_cache_size, gripper_threshold):
        import torch
        from transformers import AutoImageProcessor, AutoModel
        self.backend = backend
        self.contexts = OrderedDict()
        self.context_cache_size = context_cache_size
        self.gripper_threshold = gripper_threshold
        self.horizon = backend.actor.model.config.action_horizon
        self.action_dim = backend.actor.model.config.action_dim
        self.dino_processor = AutoImageProcessor.from_pretrained(dino_path, local_files_only=True, use_fast=False)
        self.dino = AutoModel.from_pretrained(dino_path, local_files_only=True).eval().requires_grad_(False)

    def reference_observation(self, observation):
        # This is the very same source materialization/crop path used by Dense
        # base evaluation, before TorchBackend's own norm/tokenize/resize.
        from core.app.handlers.imaging import prepare_reference_observation
        from denserl.training.online_runtime import FrozenOnlineInputs
        raw = dict(observation)
        if '_reference_image_transform' in raw:
            raw = prepare_reference_observation(raw)
        return raw, FrozenOnlineInputs.flat(raw)

    def infer(self, request):
        import torch
        from torch.utils._pytree import tree_map
        request = dict(request)
        operation = request.pop('dsrl_operation')
        if operation in ('features', 'prepare'):
            started = time.perf_counter()
            prefix = np.asarray(request.pop('rtc_prefix', []), np.float32).reshape(-1, 14)
            if len(prefix) not in (0, 5):
                raise ValueError('Dense base requires initial prefix0 or committed prefix5')
            raw, flat = self.reference_observation(request)
            observation, inputs = self.backend._transform_obs(flat)
            if self.backend.actor.compile_mode:
                torch.compiler.cudagraph_mark_step_begin()
            with torch.no_grad():
                _, mask, kv, hidden = self.backend.actor.prepare_observation(observation)
                # Compiled graph outputs may be reused on the next call. A token
                # must own its original source even across another prepare.
                kv = tree_map(lambda x: x.clone() if torch.is_tensor(x) else x, kv)
                mask = mask.clone()
                pixels = self.dino_processor(images=raw['images']['cam_left_wrist'], return_tensors='pt')
                dino = self.dino(**pixels).last_hidden_state[0, 0].numpy()
                vlm = hidden[0].float().cpu().numpy()
            prefix_feature = np.zeros((5, 14), np.float32)
            padded = None
            if len(prefix):
                normalized = self.backend.normalize_actions(prefix)
                prefix_feature[:] = normalized
                padded = torch.zeros((5, 32), device=self.backend.device)
                padded[:, :14] = torch.as_tensor(normalized, device=self.backend.device)
            features = dict(state=np.asarray(raw['state'], np.float32), dino_wrist=dino,
                            vlm_features=vlm,
                            rtc_prefix=np.r_[prefix_feature.reshape(-1), np.float32(len(prefix)/5)])
            result = dict(features=features, server_timings=dict(features_ms=(time.perf_counter()-started)*1000))
            if operation == 'prepare':
                token = uuid.uuid4().hex
                self.contexts[token] = (mask, kv, padded, len(prefix))
                while len(self.contexts) > self.context_cache_size:
                    self.contexts.popitem(last=False)
                result['context_id'] = token
            return result
        if operation != 'sample_prepared':
            raise ValueError('Decode requires a source-bound prepared context')
        mask, kv, prefix, delay = self.contexts.pop(request['dsrl_context_id'])
        started = time.perf_counter()
        result = {}
        for field, output in (('dsrl_noise', 'actions'), ('dsrl_baseline_noise', 'baseline_actions')):
            if field not in request:
                continue
            noise = np.asarray(request[field], np.float32)
            if noise.shape != (50, 32) or not np.isfinite(noise).all():
                raise ValueError('Expected finite full H50x32 noise')
            if output == 'baseline_actions' and np.array_equal(noise, request['dsrl_noise']):
                result[output] = result['actions'].copy()
                continue
            with torch.no_grad():
                plan = self.backend.actor.sample_prepared(torch.as_tensor(noise[None], device=self.backend.device),
                    prefix, mask, kv, self.backend.denoise_steps)[0, :, :14].float().cpu().numpy()
            physical = self.backend.unnormalize_actions(plan)
            for dim in (6, 13):
                physical[:, dim] = np.where(physical[:, dim] >= self.gripper_threshold, 100., 0.)
            result[output] = physical[delay:delay+8].copy()
        result['server_timings'] = dict(sample_ms=(time.perf_counter()-started)*1000)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    args = parser.parse_args()
    from process.runtime_checks import load_settings
    cfg = load_settings(args.config)
    if Path(sys.executable).resolve() != Path(cfg['runtime_python']).resolve():
        os.execv(cfg['runtime_python'], [cfg['runtime_python'], '-m', 'process.serve_frozen_vla',
                                       '--config', str(args.config.resolve())])
    if cfg['format'] != 'r1lite_dense_frozen_vla_v2':
        raise ValueError('Expected the reviewed Dense frozen runtime contract')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg['physical_gpu'])
    os.environ['EXPO_COMPILE_MODE'] = cfg['compile_mode']
    os.environ['EXPO_COMPILE_PREFIX'] = '1'
    os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'
    sys.path[:0] = [str(Path(cfg['dense_root'])/'src'), str(Path(cfg['dense_root'])/'client/src')]
    import torch
    torch.set_num_threads(cfg['torch_num_threads'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from denserl.training.torch_backend import TorchBackend
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    backend = TorchBackend(cfg['checkpoint'], assets_dir=cfg['assets_dir'], device='cuda:0',
                           delay=5, seed=cfg['seed'], denoise_steps=cfg['denoise_steps'])
    policy = NoisePolicy(backend, cfg['dino_model'], context_cache_size=cfg['context_cache_size'],
                         gripper_threshold=cfg['gripper_threshold'])
    WebsocketPolicyServer(policy, host='127.0.0.1', port=cfg['port'], metadata=dict(
        model_horizon=50, noise_dim=32, execute_horizon=8, rtc_delay=5,
        prepared_context=True, paired_prepared=True, runtime='DenseRL.TorchBackend',
        vla_checkpoint=cfg['checkpoint'], config_path=str(args.config.resolve()))).serve_forever()


if __name__ == '__main__':
    main()
