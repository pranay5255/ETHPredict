"""Reconstruct local base models and all train-only transforms from saved state."""
from copy import deepcopy

import torch

from src.training.discipline import ConstantBase, SeedEnsemble, TabularBase


def load_base_checkpoint(path, config, dataset, device='cpu'):
    # Checkpoints are trusted local outputs; sklearn extra state requires pickle.
    state = torch.load(path, map_location=device, weights_only=False)
    from src.experiments.meta_labeling_mvp import MultiHorizonLSTM
    kind = config.get('model', {}).get('base', {}).get('type', 'lstm')
    settings = config.get('model', {}).get('base', {})

    def member(part):
        horizons = len(dataset['horizon_names'])
        if kind in {'zero', 'mean'}:
            model = ConstantBase(torch.zeros(horizons), torch.full((horizons,), .5))
        elif kind in {'linear', 'tree'}:
            model = TabularBase([], [])
        else:
            model = MultiHorizonLSTM(dataset['input_size'], int(settings.get('hidden_size', 16)),
                                     int(settings.get('num_layers', 1)), float(settings.get('dropout', 0.)), horizons)
            # Historical state dictionaries predate target transforms (identity).
            part.setdefault('return_mean', model.return_mean)
            part.setdefault('return_scale', model.return_scale)
        for name in ['feature_mean', 'feature_std']:
            model.register_buffer(name, torch.empty_like(part[name]))
        model.load_state_dict(part)
        return model.to(device)

    seeds = config.get('training', {}).get('ensemble_seeds', [])
    if any(key.startswith('members.') for key in state):
        if not seeds:
            raise ValueError('Ensemble reconstruction requires recorded seeds')
        members = [member({key[len(f'members.{i}.'):]: value for key, value in state.items()
                           if key.startswith(f'members.{i}.')}) for i in range(len(seeds))]
        return SeedEnsemble(members, seeds).to(device)
    return member(deepcopy(state))
