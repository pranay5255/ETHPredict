"""Clean fork server. Never loads input data or candidate code in the parent.

Start with an isolated Python interpreter; only the short-lived, confined child
reads each request. In particular, do not fork from the scoring process, whose
memory contains future observations and (for private grading) private reports.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import random
import resource
import signal
import sys
import traceback

from ray import cloudpickle
import numpy as np
import pandas as pd
import pyarrow.parquet  # preload library imports, not data or a CUDA context
import torch
import torch._dynamo  # eager library import avoids libc nscd probes after sealing
import sklearn.metrics
import sklearn.linear_model
import sklearn.ensemble
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot  # libraries only; font discovery stays out of candidate calls

from .contracts import FitContext, ReplayFactory, read_market_data, read_replay
from .feature_packets import FeaturePacket, read_packet, write_packet
from .sandbox import allow_worker_trace, close_inherited_descriptors, kill_with_parent, protect_parent_memory, restrict_filesystem, restrict_syscalls


def _load_strategy(entry):
    specification = importlib.util.spec_from_file_location('ethpredict_candidate', entry)
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    cloudpickle.register_pickle_by_value(module)
    return module.Strategy()


def _declaration(strategy):
    result = {name: getattr(strategy, name) for name in [
        'name', 'markets', 'resolution', 'max_horizon_bars', 'lookback_bars', 'budget_tier']}
    result.update(engineered_features=getattr(strategy, 'engineered_features', []),
                  pretrained_models=getattr(strategy, 'pretrained_models', []),
                  uses_dev_lake=getattr(strategy, 'uses_dev_lake', True))
    return result


def execute(request_path, expected_parent):
    # Only trusted controller code executes before the restrictions are installed.
    allow_worker_trace()
    kill_with_parent()
    if os.getppid() != expected_parent:
        raise RuntimeError('Worker parent died during startup')
    specification = json.loads(Path(request_path).read_text())
    output = Path(specification['output'])
    output.mkdir(exist_ok=True)
    os.setsid()
    os.chdir(output)
    for number, name in [(0, '/dev/null'), (1, str(output / 'stdout.log')), (2, str(output / 'stderr.log'))]:
        handle = os.open(name, os.O_RDONLY if number == 0 else os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
        os.dup2(handle, number)
        if handle != number:
            os.close(handle)
    close_inherited_descriptors()
    os.environ.clear()
    os.environ.update(specification['environment'])
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))
    resource.setrlimit(resource.RLIMIT_FSIZE, (specification['max_output_bytes'],) * 2)
    warmup = torch.ones(1, device=specification['device'], requires_grad=True)
    warmup.square().sum().backward()
    del warmup
    if specification['device'] == 'cuda':
        # The driver names its helper threads through /proc/self/task/*/comm.
        # Initialize trusted libraries before sealing, without permitting that
        # whole proc tree to untrusted code. No candidate/data is loaded yet.
        torch.cuda.synchronize()
    restrict_syscalls()
    restrict_filesystem(read_only=specification['read_only'], writable=specification['writable'])
    sys.dont_write_bytecode = True
    # The package's __path__ already points at the frozen grader directory.
    # Never search/list the surrounding checkout, which may contain old data.
    frozen_root = str(Path(__file__).resolve().parent.parent)
    sys.path[:] = [entry for entry in sys.path if entry != frozen_root]
    sys.path.insert(0, specification['candidate'])
    seed = specification['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    if specification['operation'] in {'declare', 'fit'}:
        strategy = _load_strategy(specification['entry'])
    else:
        # Candidate pickle is executable and is intentionally loaded ONLY here.
        with open(specification['state'], 'rb') as source:
            strategy = cloudpickle.load(source)
    if specification['operation'] == 'declare':
        (output / 'declaration.json').write_text(json.dumps(_declaration(strategy), allow_nan=False))
    elif specification['operation'] == 'fit':
        replay = read_replay(Path(specification['replay'])) if specification['replay'] else None
        context = FitContext(seed, specification['device'], specification['deadline'], output,
                             ReplayFactory(replay, seed) if replay else None,
                             Path(specification['dev_lake']) if specification['dev_lake'] else None,
                             specification['train_anchor_end_ms'],
                             {key: Path(path) for key, path in specification['models'].items()})
        strategy.fit(read_market_data(Path(specification['data'])), context)
        with open(output / 'state.pkl', 'wb') as target:
            cloudpickle.dump(strategy, target, protocol=5)
    elif specification['operation'] == 'predict':
        times = pd.DatetimeIndex(pd.to_datetime(specification['decision_times_ms'], unit='ms', utc=True))
        history = read_market_data(Path(specification['data']))
        result = strategy.predict(history, times)
        if not isinstance(result, pd.DataFrame):
            raise TypeError('predict must return a pandas DataFrame')
        result.to_parquet(output / 'positions.parquet', index=False)
        if getattr(strategy,'engineered_features',[]):
            available = callable(getattr(strategy,'feature_packet',None)) and callable(getattr(strategy,'predict_features',None))
            (output/'feature_status.json').write_text(json.dumps({'available':available}))
            if available:
                packet = strategy.feature_packet(history,times)
                write_packet(packet,output/'features')
                strategy.predict_features(packet).to_parquet(output/'feature_positions.parquet',index=False)
    elif specification['operation'] == 'predict_features':
        # This operation receives no raw history or lake view. Fitted state is
        # reloaded fresh, so a validation cache cannot bypass the permutation.
        times = pd.DatetimeIndex(pd.to_datetime(specification['decision_times_ms'], unit='ms', utc=True))
        from .contracts import Declaration
        declaration = Declaration.parse(_declaration(strategy))
        packet = read_packet(Path(specification['features']),declaration,times,seed=seed,verify_noise=False)
        strategy.predict_features(packet).to_parquet(output/'positions.parquet',index=False)
    else:
        raise ValueError('Unknown worker operation')


def main():
    protect_parent_memory()
    kill_with_parent()
    if torch.cuda.is_initialized():
        raise RuntimeError('Clean worker server must never initialize CUDA')
    # No request contents or fitted object ever enter this process's memory.
    print(json.dumps({'ready': True, 'pid': os.getpid()}), flush=True)
    for line in sys.stdin:
        request_path = json.loads(line)['request_path']
        server_pid = os.getpid()
        pid = os.fork()
        if pid == 0:
            try:
                execute(request_path, server_pid)
            except BaseException:
                traceback.print_exc()
                os._exit(1)
            os._exit(0)
        print(json.dumps({'started': pid}), flush=True)
        _, status, usage = os.wait4(pid, 0)
        try:
            os.killpg(pid, signal.SIGKILL)  # no background descendant survives a call
        except ProcessLookupError:
            pass
        print(json.dumps({'finished': pid, 'returncode': os.waitstatus_to_exitcode(status),
                          'peak_rss_bytes': usage.ru_maxrss * 1024}), flush=True)


if __name__ == '__main__':
    main()
