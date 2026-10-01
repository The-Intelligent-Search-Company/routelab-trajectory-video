import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch

from football_full_backbone_data import split_guard
from football_wan_eval_common import denoise
from nfl_conditioning import prepare_bundle, validate_bundle, edit_routes
from nfl_posttrain_sampling import clip_index, expert_indices
from nfl_preprocessing.media import render_control, letterbox_transform, nearest_frames
from nfl_rollout_contract import validate_tracks, sample_tracks
from nfl_training_loss import future_velocity_loss
from wan_move_cloud_storage import publish, read_manifest
from generate import verified_pair


def tracks():
    # Artificial software fixture, never an empirical result or included media.
    xy=np.zeros((49,22,2))
    xy[:,:,0]=np.linspace(.1,.8,22)[None,:]
    xy[:,:,1]=np.linspace(.2,.7,22)[None,:]+np.arange(49)[:,None]/1000
    return {'schema':'nfl-first-frame-tracks-v1','coordinate_system':'image_normalized',
        'times_seconds':(np.arange(49)/16).tolist(), 'positions':xy.tolist(),
        'player_ids':[f'P{i:02}' for i in range(22)],
        'team_ids':[0]*11+[1]*11, 'instruction_available':np.ones((49,22),int).tolist()}


PROFILE={'width':832,'height':480,'num_frames':49,'fps':16}


def test_bundle_roundtrip_and_mutation_guard(tmp_path):
    image=tmp_path/'start.png';Image.new('RGB',(832,480),(20,60,20)).save(image)
    bundle=tmp_path/'bundle';prepare_bundle(image,tracks(),bundle,model='wan22')
    files={p.name:p.read_bytes() for p in bundle.iterdir()}
    assert validate_bundle(files)['seed']==42
    files['trajectories.json']+=b' '
    with pytest.raises(ValueError,match='checksum'):validate_bundle(files)


def test_renderer_digest_and_target_exclusion():
    t=tracks();a=np.asarray(render_control(t,PROFILE))
    assert a.shape==(49,480,832,3) and a.dtype==np.uint8
    assert np.array_equal(a,np.asarray(render_control(t,PROFILE)))
    t['future_camera_matrices']=[]
    with pytest.raises(ValueError,match='Unexpected'):validate_tracks(t)


def test_resampling_no_extrapolation_or_padding():
    t=tracks();xy,available=sample_tracks(t,np.array([0,.03125,3]))
    assert np.allclose(xy[1],(np.asarray(t['positions'])[0]+np.asarray(t['positions'])[1])/2)
    assert available.all()
    with pytest.raises(ValueError,match='exceeds'):sample_tracks(t,np.array([3.01]))
    t['instruction_available'][1][0]=0
    assert not sample_tracks(t,np.array([.03125]))[1][0,0]


def test_route_edit_preserves_origins_and_unedited_players():
    t=tracks();edited,_=edit_routes(t,{'P03':[[.3,.3],[.4,.5]]},video=PROFILE)
    a,b=np.asarray(t['positions']),np.asarray(edited['positions'])
    assert np.array_equal(a[0],b[0])
    assert np.array_equal(np.delete(a,3,axis=1),np.delete(b,3,axis=1))
    assert not np.array_equal(a[:,3],b[:,3])


def test_identity_and_geometry_rejections():
    t=tracks();t['player_ids'][1]=t['player_ids'][0]
    with pytest.raises(ValueError,match='unique'):validate_tracks(t)
    transform=letterbox_transform((800,450),(832,480))
    assert transform['resized_size']==[832,468] and transform['padding_left_top']==[0,6]
    with pytest.raises(ValueError,match='duplicate'):nearest_frames([0,.1,.2],[0,.01,.2])


def test_game_split_leakage():
    rows=[{'sample_id':'a','game_id':'1','split':'train'}, {'sample_id':'b','game_id':'1','split':'development'}]
    with pytest.raises(ValueError,match='leakage'):split_guard(rows)


def test_sampler_resume_cursor_and_experts():
    order=[clip_index(37,42,u,0,r,8,1) for u in range(5) for r in range(8)]
    assert len(set(order[:37]))==37
    resumed=[clip_index(37,42,u,0,r,8,1) for u in range(8,32) for r in range(8)]
    full=[clip_index(37,42,u,0,r,8,1) for u in range(32) for r in range(8)]
    assert resumed==full[64:]
    assert expert_indices([1,.875,.8,0],'high_noise',.875).tolist()==[0,1]
    assert expert_indices([1,.875,.8,0],'low_noise',.875).tolist()==[2,3]


def test_future_loss_mask_does_not_score_prefix():
    pred=torch.zeros((1,16,3,2,2),requires_grad=True);target=torch.zeros_like(pred)
    target[:,:,0]=100;target[:,:,1:]=2
    loss=future_velocity_loss(pred,target,1);assert loss.item()==4
    loss.backward();assert torch.count_nonzero(pred.grad[:,:,0])==0
    assert torch.count_nonzero(pred.grad[:,:,1:])>0


def test_cfg_and_expert_switch():
    class Scheduler:
        timesteps=torch.tensor([900.,800.])
        def step(self,pred,t,z):return z-pred
    conditions={'context':torch.tensor([3.]),'y':torch.tensor([0.]),'reference_latents':torch.tensor([0.])}
    switches=[]
    def fn(dit,context,**kwargs):return context
    output,counts=denoise({'high_noise':'h','low_noise':'l'},fn,Scheduler(),torch.tensor([0.]),conditions,
        torch.tensor([1.]),cfg_scale=5,boundary=.875,activate=lambda a,b:switches.append((a,b)))
    assert output.item()==-22 and counts=={'high_noise':1,'low_noise':1}
    assert switches==[(None,'high_noise'),('high_noise','low_noise')]
    with pytest.raises(ValueError,match='whitelisted'):
        denoise({},fn,Scheduler(),torch.tensor([0.]),dict(conditions,target=torch.tensor([0.])),
            torch.tensor([1.]),cfg_scale=5,boundary=.875,activate=lambda *_:None)


def test_committed_storage_detects_corruption_and_refuses_replacement(tmp_path):
    source=tmp_path/'src';source.mkdir();(source/'rank.pt').write_bytes(b'original')
    target=tmp_path/'dst';publish(source,target)
    assert read_manifest(target,True)['rank.pt']['bytes']==8
    (target/'rank.pt').write_bytes(b'corrupt!')
    with pytest.raises(ValueError,match='integrity'):read_manifest(target,True)
    with pytest.raises(ValueError):publish(source,target)


def test_metrics_phase_continuity():
    result=json.loads((Path(__file__).parents[1]/'results/measured_validation.json').read_text())
    for values in result['experts'].values():
        a,b=values['phases']
        assert (a['start_step'],a['stop_step'],b['start_step'],b['stop_step'])==(0,8,8,32)
        assert a['development_fixed_noise_loss_after']==b['development_fixed_noise_loss_before']
        assert values['relative_reduction_percent']==pytest.approx(100*(1-values['adapted_step_32']/values['pretrained_step_0']))


def test_mixed_candidate_pair_rejected(tmp_path):
    from wan_move_cloud_storage import commit_directory, digest
    for expert,identity in [('high_noise','a'*64),('low_noise','b'*64)]:
        folder=tmp_path/expert;folder.mkdir();(folder/'model.safetensors').write_bytes(b'fixture')
        (folder/'export.json').write_text(json.dumps({'identity':identity,'expert':expert,'step':32,
            'original_world_size':8,'state_sha256':digest(folder/'model.safetensors'),
            'all_rank_sha256_verified':True,'all_tensors_finite':True,'all_tensors_serialization_roundtrip_equal':True}))
        commit_directory(folder)
    with pytest.raises(ValueError,match='mismatch'):verified_pair(tmp_path,'a'*64,32)
