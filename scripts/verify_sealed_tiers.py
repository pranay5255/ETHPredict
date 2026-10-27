"""Unprivileged acceptance probe. Denial proves only the private lake boundary.

Run after the HUMAN has sealed the lake. This never uses sudo, evaluates a
strategy, or opens another copy of shadow/holdout observations.
"""
import argparse
import json
import os
from pathlib import Path

if __package__:
    from .seal_grader_tiers import read_plan, validate_plan
else:
    from seal_grader_tiers import read_plan, validate_plan


def verify_denial(plan: dict) -> dict:
    validate_plan(plan)
    if os.geteuid() == 0:
        raise ValueError('Run as the agent user, not root')
    source, private = Path(plan['source_root']), Path(plan['private_root'])
    targets = [private / 'sealed.json', private / 'seal-plan.json']
    targets += [private / move['destination'] for move in plan['moves']]
    targets += [private / 'original_manifests' / item['market'] / '_manifest.json' for item in plan['manifests']]
    targets += [private / 'lake/lighter' / item['market'] / '_manifest.json' for item in plan['manifests']]
    for path in targets:
        try:
            with path.open('rb'):
                pass
        except PermissionError:
            continue
        except FileNotFoundError as exc:
            raise ValueError('Missing private path is not evidence of permission denial') from exc
        raise ValueError(f'Agent can open a private path: {path}')
    for move in plan['moves']:
        if (source / move['source']).exists():
            raise ValueError('Private source copy remains in public lake')
    for item in plan['manifests']:
        public = json.loads((source / item['market'] / '_manifest.json').read_text())
        if public != item['public']:
            raise ValueError('Public manifest is not the planned dev-only view')
    return {'private_path_denial': 'passed', 'permission_errors': len(targets),
            'agent_uid': os.geteuid(), 'public_lake_private_copies': 0,
            'overall_agent_isolation': 'not proven: legacy CSV/Git/artifact copies and worker sandbox require separate checks'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = verify_denial(read_plan(args.plan))
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
