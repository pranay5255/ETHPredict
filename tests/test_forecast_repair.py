from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import torch

from src.experiments.meta_labeling_mvp import build_multi_horizon_lighter_dataset, train_base_model, predict_base_model
from src.training.checkpoints import load_base_checkpoint
from tests.test_staged_trial import _v2_config, _write_5m_ohlcv


@pytest.fixture
def data(tmp_path):
    root=tmp_path/'data';_write_5m_ohlcv(root/'raw'/'ETHUSDT-5m-lighter-fixture.csv',rows=180)
    cfg=_v2_config(root,tmp_path/'runs')
    cfg['training'].update(epochs=3,ensemble_seeds=[17],target_standardization='standard',
        early_stopping={'enabled':True,'validation_fraction':.2,'patience':2,'min_delta':0.})
    return build_multi_horizon_lighter_dataset(cfg),cfg


def test_target_transform_excludes_inner_holdout_and_roundtrips(data,tmp_path):
    dataset,cfg=data;train=np.arange(100)
    model,history=train_base_model(dataset,train,cfg,torch.device('cpu'))
    fit=history['early_stopping']['fit_indices'];held=history['early_stopping']['evaluation_indices']
    assert torch.allclose(model.return_mean,dataset['y_ret'][fit].mean(0))
    assert torch.allclose(model.return_scale,dataset['y_ret'][fit].std(0,unbiased=False).clamp_min(1e-8))
    changed=deepcopy(dataset);changed['y_ret'][held]=100.;changed['y_ret'][100:]=100.
    second,_=train_base_model(changed,train,cfg,torch.device('cpu'))
    assert torch.equal(model.return_mean,second.return_mean)
    assert torch.equal(model.return_scale,second.return_scale)
    # Nonzero transforms make a missing/double inverse transform visible.
    with torch.no_grad():
        model.return_mean.fill_(.002);model.return_scale.fill_(.0003)
    path=tmp_path/'base.pt';torch.save(model.state_dict(),path)
    restored=load_base_checkpoint(path,cfg,dataset)
    pd.testing.assert_frame_equal(predict_base_model(model,dataset,np.arange(120,130),torch.device('cpu')),
                                  predict_base_model(restored,dataset,np.arange(120,130),torch.device('cpu')))
    assert len(history['training_loss_components'])==history['early_stopping']['epochs_run']
    for row in history['training_loss_components']:
        assert row['combined_objective']==pytest.approx(cfg['training']['loss_weights']['return']*row['return_mse_objective']+cfg['training']['loss_weights']['direction']*row['direction_cross_entropy'],rel=1e-5)


@pytest.mark.parametrize('kind',['zero','mean','linear','lstm'])
def test_comparators_share_units_and_checkpoint_contract(data,tmp_path,kind):
    dataset,cfg=data;cfg['model']['base']['type']=kind
    model,_=train_base_model(dataset,np.arange(100),cfg,torch.device('cpu'))
    prediction=predict_base_model(model,dataset,np.arange(120,130),torch.device('cpu'))
    if kind=='zero':
        assert (prediction.next_5m_pred_return==0).all()
    if kind in {'zero','mean'}:
        assert prediction.next_5m_direction_prob.to_numpy()==pytest.approx(float(dataset['y_dir'][:100,0].mean()))
    path=tmp_path/'base.pt';torch.save(model.state_dict(),path)
    restored=load_base_checkpoint(path,cfg,dataset)
    pd.testing.assert_frame_equal(prediction,predict_base_model(restored,dataset,np.arange(120,130),torch.device('cpu')))


def test_return_target_is_forward_log_fraction_on_the_declared_clock(data):
    dataset,_=data;sample=dataset['samples'].iloc[40];i=int(sample.sample_index)
    for h,name in enumerate(dataset['horizon_names']):
        bars=dataset['horizon_bars'][name]
        expected=np.log(dataset['price_path'].close.iloc[i+bars]/dataset['price_path'].close.iloc[i])
        assert float(dataset['y_ret'][40,h])==pytest.approx(expected,rel=1e-6)
    assert int(sample.bar_close_timestamp.value//10**6)==int(dataset['price_path'].close_time.iloc[i])


def test_standardized_lstm_can_learn_controlled_small_return_signal(data):
    dataset,cfg=data
    cfg['training'].update(epochs=100,learning_rate=.03,batch_size=32,loss_weights={'return':1.,'direction':0.},
                           early_stopping={'enabled':False})
    # Known smooth input -> known small target; verifies units and learnability.
    x=torch.linspace(-1,1,32)
    dataset['X'][:32]=x[:,None,None].expand_as(dataset['X'][:32])
    dataset['y_ret'][:32]=(.001*x[:,None]).expand_as(dataset['y_ret'][:32])
    model,_=train_base_model(dataset,np.arange(32),cfg,torch.device('cpu'))
    pred=predict_base_model(model,dataset,np.arange(32),torch.device('cpu'))
    y=dataset['y_ret'][:32,0].numpy();p=pred.next_5m_pred_return.to_numpy()
    assert np.mean((p-y)**2)<.1*np.mean(y*y)
