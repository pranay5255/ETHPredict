"""Closed schemas for grader-owned report structures; no candidate extensions."""
from __future__ import annotations


def obj(properties, required=None):
    return {'type':'object','additionalProperties':False,'properties':properties,
            'required':list(properties) if required is None else required}


def array(items, **bounds):
    return {'type':'array','items':items,**bounds}


STRING = {'type':'string'}
NUMBER = {'type':'number'}
NULL_NUMBER = {'type':['number','null']}
INTEGER = {'type':'integer','minimum':0}
BOOLEAN = {'type':'boolean'}
SCORE = {'type':'number','minimum':0,'maximum':1}
SHA = {'type':'string','pattern':'^[a-f0-9]{64}$'}
STRINGS = array(STRING)
INDICES = array(INTEGER)
WINDOW = obj({'start_ms':INTEGER,'end_ms':INTEGER})
GROUPS = {'type':'object','additionalProperties':STRINGS}
HASHES = {'type':'object','additionalProperties':SHA}
SEED_NET = {'type':'object','additionalProperties':NUMBER}
COMPONENT_NAMES = ['mean_sharpe','worst_sharpe','deflated_sharpe','cost_robustness','drawdown','pnl_per_turnover']

DECLARATION = obj({'name':STRING,'markets':array(STRING,minItems=1,uniqueItems=True),
    'resolution':{'enum':['1m','5m','1h']},'max_horizon_bars':{'type':'integer','minimum':1},
    'lookback_bars':{'type':'integer','minimum':1},'budget_tier':{'enum':['fast','long']},
    'engineered_features':array(STRING,uniqueItems=True),'pretrained_models':array(STRING,uniqueItems=True),
    'uses_dev_lake':BOOLEAN})
MODEL = obj({'model':STRING,'repository':STRING,'revision':STRING,'license':STRING,
    'release_date':{'type':['string','null']},'release_evidence':STRING,'license_evidence':STRING,'files':HASHES,
    'release_safe_after_ms':{'type':['integer','null']},'period_label':STRING,
    'possibly_pretraining_window':{'oneOf':[{'type':'null'},WINDOW]},
    'holdout_release_eligible':BOOLEAN,'note':STRING})
SCRATCH_MODEL = obj({'initialization':STRING,'external_checkpoint':{'const':False},'license':STRING})
MODEL_BLOCK = obj({'block':{'type':'integer','minimum':1,'maximum':5},'training_window':WINDOW,
    'evaluation_window':WINDOW,'training':array(MODEL),'evaluation':array(MODEL)})
BASELINE = obj({'grading_fingerprint':SHA,'data_sha256':SHA,'markets':STRINGS,'resolution':STRING,
    'tier':{'enum':['dev','shadow']},'trial_count':{'type':'integer','minimum':6},
    'winner':STRING,'best_net_by_seed':SEED_NET})

STATISTIC = obj({'available':BOOLEAN,'value':{'oneOf':[NUMBER,{'const':'unavailable'}]},
    'reason':{'type':['string','null']},'path_type':STRING,'observations':INTEGER,
    'benchmark_sharpe':NUMBER,'trial_count':{'type':'integer','minimum':1},
    'trial_sharpe_variance':NUMBER,'null_mean':NUMBER},
    required=['available','value','reason','path_type'])
BLOCK = obj({'block':{'type':'integer','minimum':1,'maximum':5},'net_1x_bps':NUMBER,
    'net_2x_bps':NUMBER,'turnover':{'type':'number','minimum':0},'trades':INTEGER,'daily_sharpe':NULL_NUMBER})
METRIC = obj({'pre_gate_score':SCORE,'raw_components':obj({name:NULL_NUMBER for name in COMPONENT_NAMES}),
    'components':obj({name:SCORE for name in COMPONENT_NAMES}),'unavailable_components':STRINGS,
    'deflated_sharpe_evidence':STATISTIC,'net_1x_bps':NUMBER,'net_2x_bps':NUMBER,
    'turnover':{'type':'number','minimum':0},'trades':INTEGER,'trial_count':{'type':'integer','minimum':1},
    'trial_sharpes':array(NUMBER),'daily_sharpe_per_period':STATISTIC,'annualization':STRING,
    'seed':INTEGER,'blocks':array(BLOCK,minItems=5,maxItems=5)})
GATE_NAMES = ['enough_blocks','positive_net_expectancy','enough_trades','positive_across_blocks',
              'block_trade_coverage','survives_cost_stress','comparable_blocks','beats_best_baseline']
ECONOMICS = obj({'status':{'enum':['passed','rejected']},'passed':BOOLEAN,
    'gates':obj({name:BOOLEAN for name in [*GATE_NAMES,'credible_features']},required=GATE_NAMES),
    'positive_blocks':INTEGER,'scope':STRING,'reason':STRING,'note':STRING,'failures':STRINGS,
    'best_baseline_net_bps':NULL_NUMBER,'seed':INTEGER})

PERMUTATION = obj({'feature':STRING,'model':STRING,'method':STRING,'is_noise':BOOLEAN,
    'f1_drop':NUMBER,'negative_log_loss_drop':NUMBER,'f1_std':NUMBER,'negative_log_loss_std':NUMBER,'fold':INTEGER})
IMPORTANCE_FOLD = obj({'fold':INTEGER,'available':BOOLEAN,'reason':{'type':['string','null']},
    'permutation_rows':array(PERMUTATION),'auxiliary_comparisons':INTEGER,'split_audit':{'type':'object'},
    'fit_indices':INDICES,'evaluation_indices':INDICES,'scored_policy_rows':INDICES,'nonexecuted_rows':INDICES,
    'families':GROUPS,'noise':GROUPS,'baseline':obj({'f1':NULL_NUMBER,'negative_log_loss':NULL_NUMBER}),
    'target':STRING,'probability_proxy':STRING,'permutation_scope':STRING},
    required=['fold','available','reason','permutation_rows','auxiliary_comparisons','split_audit'])
IMPORTANCE_SUMMARY = obj({'model':STRING,'feature':STRING,'negative_log_loss_drop':NUMBER,'fold_std':NUMBER,
    'f1_drop':NUMBER,'folds':INTEGER,'category':STRING})
IMPORTANCE_REPORT = obj({'status':{'enum':['completed','unavailable']},'scope':STRING,
    'folds':array(IMPORTANCE_FOLD,maxItems=5),'summary':array(IMPORTANCE_SUMMARY),'noise_control_failures':STRINGS,
    'auxiliary_comparisons':INTEGER,'interpretation':STRING,'method':STRING,'unavailable_folds':array(IMPORTANCE_FOLD)},
    required=['status','scope','folds','summary','noise_control_failures','auxiliary_comparisons','interpretation','method'])

AUDIT = obj({'opened':INTEGER,'violations':STRINGS,'complete':BOOLEAN})
CALL = obj({'operation':{'enum':['declare','fit','predict','predict_features']},
    'status':{'enum':['completed','rejected','timeout','nondeterministic']},
    'duration_seconds':{'type':'number','minimum':0},'peak_rss_bytes':INTEGER,'peak_vram_bytes':INTEGER,'audit':AUDIT})
RESOURCES = obj({'wall_seconds':{'type':'number','minimum':0},
    'wall_limit_seconds':{'type':'number','exclusiveMinimum':0,'maximum':3600.},
    'device':{'enum':['cpu','cuda']},'vram_reservation':{'oneOf':[{'type':'null'},obj({
        'pid':{'type':'integer','minimum':1},'total_bytes':INTEGER,'reserved_bytes':INTEGER,
        'cap_bytes':{'const':22*1000**3}})]},'peak_rss_bytes':INTEGER,'peak_vram_bytes':INTEGER,
    'calls':array(CALL),'scope':STRING})
MANIFEST = obj({'path':{'const':'artifact_manifest.json'},'sha256':SHA,'files':INTEGER,'bytes':INTEGER})
UNCERTAINTY = obj({'standard_error':{'type':'number','minimum':0},'resamples':{'const':1000},
    'expected_block_days':{'type':'number','minimum':1},'seed':INTEGER,'method':STRING,
    'mean':SCORE,'minimum':SCORE,'maximum':SCORE})

SCHEMA = obj({'version':{'const':1},'trial_id':{'type':'string','minLength':1},
    'mode':{'enum':['production','baseline_calibration','fixture']},'tier':{'enum':['dev','shadow']},
    'status':{'enum':['accepted','rejected','nondeterministic','timeout']},'score':SCORE,'pre_gate_score':SCORE,
    'started_at_utc':STRING,'finished_at_utc':STRING,
    'grader':obj({'tag':{'const':'grader-v1'},'calibration':{'enum':['pending_baseline_runs','complete']},
        'baseline':BASELINE},required=['tag','calibration']),
    # Early failures may leave objects partially populated. Check-specific
    # details and provenance have variable keys from frozen trusted producers.
    'strategy':obj({'source':obj({'sha256':SHA,'files':HASHES}),
        'pre_run_hypothesis':{'type':['string','null']},'declaration':DECLARATION,
        'models':array({'oneOf':[MODEL,SCRATCH_MODEL]}),'models_by_block':array(MODEL_BLOCK,maxItems=5)},required=[]),
    'data':obj({'sha256':SHA,'tier':{'enum':['dev','shadow']},'start_ms':INTEGER,'end_ms':INTEGER,
        'resolution':STRING,'markets':STRINGS,'provenance':{'type':'object'},
        'input_provenance':{'type':'object'},'execution_tail':STRING},required=[]),
    'seeds':array(INTEGER,minItems=2,maxItems=2,uniqueItems=True),
    'integrity':array(obj({'name':STRING,'passed':BOOLEAN,'details':{'type':'object'}})),
    'economics':array(ECONOMICS,maxItems=2),'metrics':array(METRIC,maxItems=2),
    'feature_importance':array(obj({'seed':INTEGER,'report':IMPORTANCE_REPORT}),maxItems=2),
    'resources':{'oneOf':[obj({}),RESOURCES]},'uncertainty':{'oneOf':[{'type':'null'},UNCERTAINTY]},
    'failures':array(obj({'check':STRING,'message':STRING})),
    'artifacts':obj({'seed_blocks':array(array(STRING,maxItems=5),maxItems=2),
        'manifest':{'oneOf':[{'type':'null'},MANIFEST]}}),
    'qualitative':obj({'status':{'const':'not_run'},'tie_breaker_only':{'const':True}})})

# Synthetic sleeper fixtures use a short internal deadline. Production and
# calibration reports must still use exactly their predeclared tier budget.
SCHEMA['allOf'] = [
    {'if':{'properties':{'mode':{'enum':['production','baseline_calibration']},
        'strategy':{'required':['declaration'],'properties':{'declaration':{
            'required':['budget_tier'],'properties':{'budget_tier':{'const':tier}}}}}}},
     'then':{'properties':{'resources':{'required':['wall_limit_seconds'],
        'properties':{'wall_limit_seconds':{'const':seconds}}}}}}
    for tier,seconds in [('fast',600.),('long',3600.)]
]
