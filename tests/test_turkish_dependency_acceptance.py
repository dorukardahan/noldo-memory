"""Offline acceptance through the real API, with both former imports forbidden."""
import json
import os
from pathlib import Path
import subprocess
import sys


def test_turkish_capture_index_recall_without_nlp_imports(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    env = {'PATH': os.defpath, 'HOME': str(tmp_path), 'TMPDIR': str(tmp_path),
           'LANG': 'en_US.UTF-8', 'AGENT_MEMORY_DATA_DIR': str(tmp_path),
           'PYTHONDONTWRITEBYTECODE': '1'}
    result = subprocess.run([sys.executable, str(repo / 'scripts/evaluate_turkish_dependency.py'), '--block-nlp'],
                            cwd=tmp_path, env=env, capture_output=True, text=True, check=True, timeout=30)
    report = json.loads(result.stdout)
    assert report['captured'] == 6
    assert report['nlp_import_attempts'] == report['loaded_nlp'] == []
    assert len(report['cases']) == 8
    assert all(row['ranked'][0] == row['expected'] for row in report['cases'])
    assert all(row['source_sessions'] == ['session-a'] for row in report['cases'])
    assert all(row['count'] == 0 for row in report['unrelated_at_floor_0_5'])
    baseline = json.loads((repo / 'docs/turkish-dependency-results-2026-09-09.json').read_text())['baseline']
    assert report['cases'] == baseline['cases']
