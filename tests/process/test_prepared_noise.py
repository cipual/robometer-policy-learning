"""Source-bound cache ownership and exact Dense RTC prefix slicing, without a VLA."""
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
import torch
from types import SimpleNamespace
from process.serve_frozen_vla import NoisePolicy

class Dino:
    def eval(self):return self
    def requires_grad_(self,flag):return self
    def __call__(self,**kwargs):return SimpleNamespace(last_hidden_state=torch.zeros(1,1,384))

class Dino:
    def eval(self):return self
    def requires_grad_(self,value):return self
    def __call__(self,**kwargs):return SimpleNamespace(last_hidden_state=torch.zeros(1,1,384))
class Actor:
    compile_mode=None
    model=SimpleNamespace(config=SimpleNamespace(action_horizon=50,action_dim=32))
    def prepare_observation(self,observation):return observation['state'],torch.ones(1,2,dtype=torch.bool),torch.tensor([observation['state'][0]]),torch.zeros(1,2048)
    def sample_prepared(self,noise,prefix,mask,kv,steps):return noise+kv[0]
class Backend:
    actor=Actor();device='cpu';denoise_steps=10
    def _transform_obs(self,obs):return obs,obs
    def normalize_actions(self,p):return p
    def unnormalize_actions(self,p):return p.copy()

def test_contexts_own_sources_and_consume_once_with_exact_prefix_slice():
    with patch('transformers.AutoModel.from_pretrained',return_value=Dino()),patch('transformers.AutoImageProcessor.from_pretrained',return_value=lambda **kw:{}):
        policy=NoisePolicy(Backend(),'fixture',context_cache_size=4,gripper_threshold=80)
    policy.reference_observation=lambda obs:(obs,obs)
    def prepare(state,prefix):return policy.infer(dict(dsrl_operation='prepare',state=np.full(14,state,np.float32),images=dict(cam_left_wrist=np.zeros((8,8,3),np.uint8)),rtc_prefix=prefix))
    first=prepare(1,[]);second=prepare(2,np.zeros((5,14),np.float32));noise=np.arange(50*32,dtype=np.float32).reshape(50,32)/100
    def sample(token):return policy.infer(dict(dsrl_operation='sample_prepared',dsrl_context_id=token,dsrl_noise=noise,dsrl_baseline_noise=np.zeros((50,32),np.float32)))
    a=sample(first['context_id']);b=sample(second['context_id'])
    arms=[0,1,2,3,4,5,7,8,9,10,11,12]
    np.testing.assert_array_equal(a['actions'][:,arms],(noise[:8,:14]+1)[:,arms])
    np.testing.assert_array_equal(b['actions'][:,arms],(noise[5:13,:14]+2)[:,arms])
    assert second['features']['rtc_prefix'][-1]==1
    with pytest.raises(KeyError):sample(first['context_id'])
