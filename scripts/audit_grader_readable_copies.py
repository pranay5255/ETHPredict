"""Inventory possible tier bypasses without reading market values or outcomes.

The monthly seal alone cannot remove historical CSV/Git/artifact copies. This
read-only inventory makes that limitation reviewable; it never deletes history.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess


def inventory(repo: Path) -> dict:
    tracked = subprocess.check_output(['git','-C',str(repo),'ls-files','data'],text=True).splitlines()
    sources=[]
    for name in tracked:
        if not name.endswith(('.csv','.parquet','.json')) or name=='data/schema.json':
            continue
        dates=re.findall(r'(20\d{6})',Path(name).name.replace('-',''))
        if len(dates)>=2 and max(dates) <= '20260501':
            continue
        sources.append({'file':name,'readable':(repo/name).is_file(),
                        'reason':'overlaps private periods or has no bounded timestamp contract'})
    worktrees=subprocess.check_output(['git','-C',str(repo),'worktree','list','--porcelain'],text=True)
    checkouts=[line.removeprefix('worktree ') for line in worktrees.splitlines() if line.startswith('worktree ')]
    common=subprocess.check_output(['git','-C',str(repo),'rev-parse','--path-format=absolute','--git-common-dir'],text=True).strip()
    derived=[]
    for name in ['artifacts','data/processed_features','results']:
        path=repo/name
        if path.exists():
            derived.append({'directory':name,'files_requiring_review':sum(p.is_file() for p in path.rglob('*'))})
    receipt=repo/'context/funding_gap_recovery_20261027.json'
    return {'scope':'Path inventory only; no source values or strategy outcomes read',
            'tracked_sources_requiring_restriction':sources,'worktrees_requiring_agent_view_review':checkouts,
            'readable_git_object_store':common if Path(common).is_dir() else None,
            'raw_funding_recovery_receipt_in_git':str(receipt) if receipt.is_file() else None,
            'derived_directories_requiring_review':derived,
            'isolation_status':'not established while legacy files/history remain agent-readable',
            'required_action':'Human must provision a dev-only restricted agent view without private-period CSVs, raw recovery receipts, derived outputs or historical Git objects; restrict network/raw-source access for hidden grading. Preserve originals privately rather than rewriting historical evidence.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    result=inventory(args.repo)
    if args.output: args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
