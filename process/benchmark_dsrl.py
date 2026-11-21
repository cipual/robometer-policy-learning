"""Measure two production-shape SAC updates and resume on synthetic replay.

Input: --config with the actual learner configuration, GPU, seed and debug output.
Output: finite losses, exact resume comparison, timings, peak memory and checkpoint
size. No robot, reward service or online trajectory is created.
"""

import argparse
import copy
import json
import os
from pathlib import Path
import random
import subprocess
import threading
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    cfg = json.loads(parser.parse_args().config.read_text())
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg['physical_gpu'])
    import torch
    from process.r1lite_dsrl import R1LiteDSRL

    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    torch.set_num_threads(cfg['torch_num_threads'])
    config = cfg['learner']
    def unused(*args, **kwargs):
        raise AssertionError('The benchmark must not call a robot or a frozen service')
    observations = {key: np.zeros(size, np.float32) for key, size in config['features'].items()}
    output = Path(cfg['output'])
    output.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    gpu_samples = []
    stop = threading.Event()
    def monitor():
        while not stop.is_set():
            gpu_samples.append(subprocess.check_output([
                'nvidia-smi', f'--id={cfg["physical_gpu"]}',
                '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip())
            stop.wait(.2)
    observer = threading.Thread(target=monitor, daemon=True)
    observer.start()
    with R1LiteDSRL(unused, None, config, 'cuda:0') as agent:
        for index in range(config['sac']['batch_size']):
            agent.buffer.add(copy.deepcopy(observations),
                             np.zeros((config['model_horizon'], config['noise_dim']), np.float32),
                             -.5, copy.deepcopy(observations), False, False,
                             episode_id='engineering-fixture', step_in_episode=index)
        metrics, times = [], []
        for iteration in range(2):
            torch.cuda.synchronize()
            start = time.perf_counter()
            values = {key: float(value) for key, value in agent.train_after_episode(
                config['sac']['learning_starts']).items()}
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
            if not values or not np.isfinite(list(values.values())).all():
                raise ValueError('The production-shape SAC update is not finite')
            metrics.append(values)
            if iteration == 0:
                start = time.perf_counter()
                agent.save_checkpoint(output / 'ckpt/engineering-resume.pt', environment_steps=0, episodes=0,
                                      metadata={'scope': 'synthetic benchmark only; not an online initialization'})
                save_seconds = time.perf_counter() - start
        expected = {key: value.detach().clone() for key, value in agent.algorithm.actor.state_dict().items()}
        with R1LiteDSRL(unused, None, config, 'cuda:0') as restored:
            counters = restored.load_checkpoint(output / 'ckpt/engineering-resume.pt')
            repeated = {key: float(value) for key, value in restored.train_after_episode(
                config['sac']['learning_starts']).items()}
            for key, value in restored.algorithm.actor.state_dict().items():
                torch.testing.assert_close(value, expected[key], rtol=0, atol=cfg['resume_atol'])
            for key, value in metrics[-1].items():
                if 'loss' in key:
                    np.testing.assert_allclose(repeated[key], value, rtol=0, atol=cfg['resume_atol'])
        report = dict(format='dsrl_production_shape_benchmark_v1', scope='synthetic replay, no robot',
                      physical_gpu=cfg['physical_gpu'], iterations=2, repeat_update_after_resume=True,
                      resume_counters=counters, update_seconds=times, metrics=metrics,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      checkpoint_bytes=(output / 'ckpt/engineering-resume.pt').stat().st_size,
                      checkpoint_save_seconds=save_seconds,
                      gpu_samples_mib_percent=gpu_samples,
                      gpu_snapshot=subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                                            '--format=csv'], text=True))
        (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(output / 'report.json', flush=True)
    stop.set()
    observer.join(timeout=5)


if __name__ == '__main__':
    main()
