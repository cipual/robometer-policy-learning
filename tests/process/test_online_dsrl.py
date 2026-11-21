"""Noise-decision clock, factual execution, manual labels and exact resume; no ROS."""
import json,time,threading
from pathlib import Path
import numpy as np
import pytest
import torch
from process.online_dsrl import OnlineDSRL,materialize

class Frozen:
    def __init__(self):
        self.contexts={}
        self.entered=threading.Event();self.release=None
    def features(self,obs,*,prefix=None):
        obs=materialize(obs);p=np.zeros(71,np.float32)
        if prefix is not None and len(prefix):p[:70]=np.asarray(prefix).reshape(-1);p[-1]=1
        return dict(state=obs['state'],vlm_features=np.zeros(16,np.float32),rtc_prefix=p)
    def prepare(self,obs,*,prefix=None):
        self.entered.set()
        if self.release is not None:assert self.release.wait(3)
        token=str(len(self.contexts));self.contexts[token]=(materialize(obs),len(prefix) if prefix is not None else 0)
        return self.features(obs,prefix=prefix),token
    def infer(self,obs,*,noise,baseline_noise):
        obs,delay=self.contexts.pop(obs['dsrl_context_id'])
        assert noise.shape==baseline_noise.shape==(50,32)
        def actions(value):
            a=obs['state'][None]+value[delay:delay+8,:14]*.01;a[:,[6,13]]=100;return a
        return dict(actions=actions(noise),baseline_actions=actions(baseline_noise))

class Reward:
    def predict(self,frames,indices,task,episode_id):
        return np.full(len(indices),.25,np.float32),np.zeros(len(indices),np.float32)

def observation(step):
    t=100+step/15;s=np.zeros(14,np.float32);s[[6,13]]=100
    return dict(state=s,images={k:np.full((8,8,3),step,np.uint8) for k in ('cam_high','cam_left_wrist','cam_right_wrist')},prompt='test',_residual_capture_time=t,_image_source_times={k:t for k in ('cam_high','cam_left_wrist','cam_right_wrist')})

def settings(path):
    learner=dict(features=dict(state=14,vlm_features=16,rtc_prefix=71),capacity=16,noise_bound=2.,noise_dim=32,chunk_size=8,model_horizon=50,warmup_seed=42,success_threshold=None,step_penalty=1.,success_bonus=1.,reward_batch_size=2,
      actor=dict(d_model=32,nhead=4,num_encoder_layers=1,feature_hidden_dims=[32],transformer_dropout=0.),
      critic=dict(d_model=32,nhead=4,num_encoder_layers=1,feature_hidden_dims=[32],action_embedding_dim=32,pooling_strategy='first',transformer_dropout=0.),
      sac=dict(gamma=.995**8,compute_chunked_gamma=False,learning_starts=0,batch_size=2,target_entropy=0,num_critic_updates_per_actor_update=1,train_actor_with_kl_divergence=False))
    return dict(output=str(path),learner=learner,task='test',max_episode_seconds=60,action_control=dict(gripper_source='frozen_vla'),outcome_max_score=1,trajectories=1,max_episode_steps=16,max_source_gap_s=.25,gripper_threshold=80,save_timeout_s=2,recording_manifest=str(path/'native.jsonl'),updates_per_transition=1,log_every=1)

def begin(session):return session.request(dict(kind='begin',task='test',recording_episode_index=0))['episode_id']
def observe(session,episode,step):session.request(dict(kind='observe',episode_id=episode,observation=observation(step)))
def generate(session,episode,step,prefix):return session.request(dict(kind='prefetch',episode_id=episode,source_step=step,source_time=observation(step)['_residual_capture_time'],prefix=prefix))
def activate(session,episode,bundle):return session.request(dict(kind='activate',episode_id=episode,proposal_id=bundle['proposal_id']))
def published(session,episode,chunk,step):
    row=step%8;a=chunk['actions'][row].copy();a[[6,13]]=100
    return session.request(dict(kind='published',episode_id=episode,chunk_id=chunk['chunk_id'],row=row,source_time=observation(step)['_residual_capture_time'],action=a,publication_completed_at=observation(step)['_residual_capture_time']+.001))

def collect(session):
    episode=begin(session);future=None;chunk=None
    for step in range(16):
        observe(session,episode,step)
        if step==0:chunk=activate(session,episode,generate(session,episode,0,np.empty((0,14),np.float32)))
        elif step%8==0:chunk=activate(session,episode,future)
        if step%8==3:future=generate(session,episode,step,chunk['actions'][3:8])
        published(session,episode,chunk,step)
    session.request(dict(kind='end',episode_id=episode,observation=observation(16),prefix=[],reason='operator_stop'))
    return episode

def test_rtc_decisions_train_only_after_actual_publications_saved_manual_outcome_and_resume(tmp_path):
    cfg=settings(tmp_path);session=OnlineDSRL(cfg,Frozen(),Reward(),device='cpu')
    try:
        episode=collect(session);ep=session.episodes[0]
        assert [x.frame_indices for x in ep.transitions]==[list(range(4,12)),list(range(12,17))]
        assert session.agent.buffer.size()==0
        session.request(dict(kind='score',recording_episode_index=0,fields=dict(score=0,max_score=1)))
        time.sleep(.05);assert session.agent.buffer.size()==0
        Path(cfg['recording_manifest']).write_text(json.dumps(dict(episode_index=0,policy_episode_id=episode,length=16,score=0,max_score=1))+'\n')
        session.worker.join(20)
        assert session.training_error is None and session.completed==1
        assert session.agent.buffer.size()==2 and session.agent.algorithm._n_updates==2
        assert len(list((tmp_path/'replay'/episode).glob('step-*.npz')))==16
        restored=OnlineDSRL(cfg,Frozen(),Reward(),device='cpu')
        try:
            restored.resume(tmp_path/'ckpt/episode-0001.pt')
            assert restored.completed==1 and restored.environment_steps==16
            assert restored.agent.buffer.size()==2 and restored.agent.algorithm._n_updates==2
            assert torch.equal(restored.agent.prior_rng.get_state(),session.agent.prior_rng.get_state())
        finally:restored.close()
        with pytest.raises(ValueError,match='budget'):begin(session)
    finally:session.close()

def test_warmup_uses_exact_dense_gaussian_prior_without_calling_actor(tmp_path):
    cfg=settings(tmp_path);cfg['learner']['sac']['learning_starts']=256
    session=OnlineDSRL(cfg,Frozen(),Reward(),device='cpu')
    def forbidden(*args,**kwargs):raise AssertionError('warmup called actor')
    session.agent.algorithm.actor.act=forbidden
    try:
        episode=begin(session);observe(session,episode,0);bundle=generate(session,episode,0,[])
        session._check_io(session.active,wait=True)
        with np.load(tmp_path/'replay'/episode/f"chunk-{bundle['proposal_id']}.npz") as x:
            gen=torch.Generator(device='cpu').manual_seed(42);expected=torch.randn((50,32),generator=gen).numpy()
            np.testing.assert_array_equal(x['noise']*2,expected)
            np.testing.assert_array_equal(x['baseline_noise'],expected)
            np.testing.assert_array_equal(x['actions'],x['baseline_actions'])
            assert x['noise_source'].item()=='base_gaussian_warmup'
    finally:session.close()


def test_initial_preparation_has_no_executed_edge_and_uses_fresh_first_source(tmp_path):
    cfg=settings(tmp_path);session=OnlineDSRL(cfg,Frozen(),Reward(),device='cpu')
    try:
        episode=begin(session);observe(session,episode,0)
        bundle=generate(session,episode,0,[])
        fresh=observation(0);fresh['_residual_capture_time']+=.4
        fresh['_image_source_times']={k:fresh['_residual_capture_time'] for k in fresh['images']}
        fresh['images']={k:np.full_like(v,7) for k,v in fresh['images'].items()}
        session.request(dict(kind='observe',episode_id=episode,observation=fresh))
        chunk=activate(session,episode,bundle)
        assert session.active.published==0 and len(session.active.frames)==1
        assert np.all(session.active.frames[0]==7) and not session.active.transitions
        session.request(dict(kind='published',episode_id=episode,chunk_id=chunk['chunk_id'],row=0,
            source_time=fresh['_residual_capture_time'],action=chunk['actions'][0],publication_completed_at=101.))
        late=dict(fresh,_residual_capture_time=101.)
        with pytest.raises(ValueError,match='Discontinuous'):
            session.request(dict(kind='observe',episode_id=episode,observation=late))
    finally:session.close()

def test_slow_prefetch_does_not_hold_the_control_observe_publish_lock(tmp_path):
    frozen=Frozen();session=OnlineDSRL(settings(tmp_path),frozen,Reward(),device='cpu')
    try:
        episode=begin(session);observe(session,episode,0);chunk=activate(session,episode,generate(session,episode,0,[]))
        for step in range(3):
            if step:observe(session,episode,step)
            published(session,episode,chunk,step)
        observe(session,episode,3);frozen.entered.clear();frozen.release=threading.Event();errors=[]
        def work():
            try:generate(session,episode,3,chunk['actions'][3:8])
            except Exception as e:errors.append(e)
        worker=threading.Thread(target=work);worker.start();assert frozen.entered.wait(2)
        start=time.perf_counter();published(session,episode,chunk,3);observe(session,episode,4)
        assert time.perf_counter()-start<.1
        frozen.release.set();worker.join(3);assert not errors
    finally:
        if frozen.release:frozen.release.set()
        session.close()

def test_no_credit_for_missing_or_modified_publication_and_source_gaps(tmp_path):
    session=OnlineDSRL(settings(tmp_path),Frozen(),Reward(),device='cpu')
    try:
        episode=begin(session);observe(session,episode,0);chunk=activate(session,episode,generate(session,episode,0,[]))
        with pytest.raises(ValueError,match='source'):
            session.request(dict(kind='published',episode_id=episode,chunk_id=chunk['chunk_id'],row=0,source_time=99,action=chunk['actions'][0],publication_completed_at=100.001))
        assert session.agent.buffer.size()==0
    finally:session.close()


def test_native_rtc_rpc_and_three_video_roundtrip(tmp_path):
    import os,subprocess
    from openpi_client import msgpack_numpy
    from websockets.sync.server import serve
    cfg=settings(tmp_path);cfg['save_timeout_s']=15;cfg['recording_manifest']=str(tmp_path/'native/meta/episodes.jsonl')
    learner=OnlineDSRL(cfg,Frozen(),Reward(),device='cpu')
    def handler(connection):
        packer=msgpack_numpy.Packer()
        connection.send(packer.pack(dict(control_hz=15,rtc_delay=5,max_source_gap_s=.25,actor_deadline_s=1/3)))
        for data in connection:
            try:reply=learner.request(msgpack_numpy.unpackb(data))
            except Exception as error:reply=dict(ok=False,error=str(error))
            connection.send(packer.pack(reply))
    root=Path(__file__).resolve().parents[2];native=root.parent/'DenseRL'
    try:
        with serve(handler,'127.0.0.1',0,max_size=None,compression=None) as server:
            worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
            env=dict(os.environ,PYTHONPATH=f'{native}/client/src:{root}',OMP_NUM_THREADS='1')
            result=subprocess.run([str(native/'.venv/bin/python'),str(root/'tests/process/native_dsrl_roundtrip.py'),str(server.socket.getsockname()[1]),str(tmp_path/'native')],env=env,timeout=40,capture_output=True,text=True)
            server.shutdown();worker.join(3)
            assert result.returncode==0,result.stdout+result.stderr
        learner.worker.join(15)
        assert learner.training_error is None and learner.completed==1
        assert learner.agent.buffer.size()==2
        assert len(list((tmp_path/'native').glob('videos/chunk-000/*/episode_000000.mp4')))==3
    finally:learner.close()
