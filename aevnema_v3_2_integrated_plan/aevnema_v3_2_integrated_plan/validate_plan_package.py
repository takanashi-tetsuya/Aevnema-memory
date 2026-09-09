#!/usr/bin/env python3
"""Validate the plan package, not the Aevnema implementation or its experiments.

Standard library only; no network, no repository/database writes.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent
MAIN = 'Aevnema_v3_2_integrated_development_experiment_plan_2026-09-08.md'
REQUIRED = (MAIN, 'README.md', 'AGENT_START_HERE.md', 'execution_backlog.json',
            'experiment_program.template.json', 'campaign_state.template.json',
            'source_evidence_index.json', 'validate_plan_package.py')

class PlanError(ValueError):
    pass

def require(test: bool, message: str) -> None:
    if not test:
        raise PlanError(message)

def load(name: str) -> Any:
    with (ROOT / name).open(encoding='utf-8') as f:
        return json.load(f)

def validate_objects(backlog: dict[str, Any], program: dict[str, Any],
                     state: dict[str, Any], sources: dict[str, Any], main: str) -> list[str]:
    tasks = backlog['tasks']
    experiments = program['experiments']
    ids = [t['id'] for t in tasks]
    eids = [e['id'] for e in experiments]
    require(len(ids) == len(set(ids)), 'Duplicate task IDs')
    require(len(eids) == len(set(eids)), 'Duplicate experiment IDs')
    require(ids == [f'W{i:02}' for i in range(20)], 'Expected W00-W19 work packages')
    require(eids == [f'E32-{i:02}' for i in range(11)], 'Expected E32-00..10 experiments')
    mapping = {t['id']: t for t in tasks}
    visited: set[str] = set()
    active: set[str] = set()
    def visit(node: str) -> None:
        require(node in mapping, f'Unknown task dependency: {node}')
        require(node not in active, f'Dependency cycle at {node}')
        if node in visited:
            return
        active.add(node)
        for dep in mapping[node]['depends_on']:
            visit(dep)
        active.remove(node)
        visited.add(node)
    for t in tasks:
        visit(t['id'])
        require(t['status'] == 'planned', 'Plan incorrectly contains execution status')
        for key in ('files', 'steps', 'regression_tests', 'acceptance', 'required_artifacts', 'on_failure', 'continuation'):
            require(bool(t.get(key)), f'{t["id"]}: missing {key}')
        require(set(t['experiment_ids']) <= set(eids), f'{t["id"]}: unknown experiment')
        require(t['id'] in main, f'{t["id"]}: missing from main plan')
    for e in experiments:
        require(set(e['work_packages']) <= set(ids), f'{e["id"]}: unknown work package')
        require(e['status'] == 'planned_not_executed', 'Experiment status is not prospective')
        require(e['expected_outcome'] == 'unknown_until_run', 'Experiment has fabricated expected measurement')
        require(e['id'] in main, f'{e["id"]}: missing from main plan')
    require(program['executable'] is False, 'A specification must not claim to be an executable runner')
    require(program['network_policy']['enabled'] is False, 'Template cannot create network authority')
    require(program['network_policy']['authorization_reference'] is None, 'Authorization must not be fabricated')
    require(program['network_policy']['per_edge_external_calls_max'] == 0, 'Per-edge external calls enabled')
    require(program['network_policy']['campaign_http_remaining'] is None, 'Do not invent a remaining budget')
    require(program['formal_scoring']['enabled'] is False, 'Formal scoring cannot be preapproved')
    require(all(program['production_actions'][x] is False for x in ('promotion','canary','formal_kb_writes')), 'Production writes/actions cannot be preauthorized')
    require(program['evidence_policy']['answer_cache'] is False, 'Answer cache cannot substitute for evidence reuse')
    require(program['evidence_policy']['gold_runtime_access'] is False, 'Runtime gold leak enabled')
    require(program['evidence_policy']['source_bound_is_semantic_entailment'] is False, 'Source binding conflated with entailment')
    require(state['status'] == 'template_not_started', 'State template is mislabeled as real work')
    require([t['id'] for t in state['tasks']] == ids, 'State task IDs differ')
    require([e['id'] for e in state['experiments']] == eids, 'State experiment IDs differ')
    source_ids = {s['id'] for s in sources['sources']}
    refs = set(re.findall(r'\[(S\d{2})\]', main))
    require(refs == source_ids, 'Source IDs in main and source index differ')
    for s in sources['sources']:
        for f in s['files']:
            require(bool(re.fullmatch(r'[a-f0-9]{64}', f['sha256'])), 'Invalid source SHA256')
            require(f['included_in_plan_bundle'] is False, 'Raw source/logs unexpectedly packaged')
    return ['required_work_package_fields', 'acyclic_dependencies', 'task_experiment_cross_references',
            'prospective_status_only', 'no_fabricated_permissions_or_measurements',
            'no_per_edge_calls_or_runtime_gold', 'source_reference_consistency', 'continuation_state_consistency']

def verify_checksums() -> str:
    manifest = ROOT / 'SHA256SUMS'
    if not manifest.exists():
        return 'not_present_during_initial_assembly'
    expected: set[str] = set()
    for line in manifest.read_text(encoding='utf-8').splitlines():
        digest, name = line.split('  ', 1)
        path = (ROOT / name).resolve()
        require(path.is_relative_to(ROOT), 'Checksum path escapes package')
        require(path.is_file(), f'Missing checksum file: {name}')
        require(hashlib.sha256(path.read_bytes()).hexdigest() == digest, f'Checksum mismatch: {name}')
        require(name not in expected, f'Duplicate checksum filename: {name}')
        expected.add(name)
    actual = {str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file() and p.name != 'SHA256SUMS' and '__pycache__' not in p.parts}
    require(expected == actual, 'Checksum manifest coverage differs from package contents')
    return f'verified_{len(expected)}_files'

def main_cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true', help='also check that intentional invalid plan variants are rejected')
    args = parser.parse_args()
    try:
        for name in REQUIRED:
            require((ROOT / name).is_file(), f'Missing required plan artifact: {name}')
        b,p,s,r = (load(x) for x in ('execution_backlog.json','experiment_program.template.json','campaign_state.template.json','source_evidence_index.json'))
        main_text = (ROOT / MAIN).read_text(encoding='utf-8')
        checks = validate_objects(b,p,s,r,main_text)
        self_tests=[]
        if args.self_test:
            variants=[]
            bad=copy.deepcopy(b); bad['tasks'][0]['depends_on']=['W01']; variants.append(('cycle_rejected',bad,p))
            bad=copy.deepcopy(b); bad['tasks'][1]['depends_on']=['W99']; variants.append(('unknown_dependency_rejected',bad,p))
            bad=copy.deepcopy(p); bad['network_policy']['enabled']=True; variants.append(('fabricated_network_authority_rejected',b,bad))
            bad=copy.deepcopy(p); bad['evidence_policy']['gold_runtime_access']=True; variants.append(('runtime_gold_leak_rejected',b,bad))
            for title,bv,pv in variants:
                try:
                    validate_objects(bv,pv,s,r,main_text)
                except PlanError:
                    self_tests.append(title)
                else:
                    raise PlanError(f'Negative self-test did not fail: {title}')
        result={'status':'passed','scope':'plan_package_only_not_repository_or_benchmark',
                'tasks':len(b['tasks']),'experiments':len(p['experiments']),'checks':checks,
                'negative_self_tests':self_tests,'checksums':verify_checksums(),
                'network_calls':0,'repository_tests_run':False,'formal_gold_approved':False}
        print(json.dumps(result,ensure_ascii=False,indent=2))
        return 0
    except (OSError,ValueError,KeyError,TypeError) as exc:
        print(json.dumps({'status':'failed','scope':'plan_package_only','error':str(exc)},ensure_ascii=False), file=sys.stderr)
        return 1

if __name__ == '__main__':
    raise SystemExit(main_cli())
