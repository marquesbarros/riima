#!/usr/bin/env python3
"""Print a deterministic manifest of selected source bytes, without private inputs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import subprocess

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_share import ROOT, source_manifest


def git_identification(root: Path) -> dict:
    """Measure HEAD and tracked/untracked changes, or report an unknown state.

    A source archive nested in another repository must never inherit its HEAD.
    No absolute paths, status filenames or remote URLs enter the inventory.
    """
    unknown = dict(git_head=None, working_tree_modified=None, git_status='unavailable')
    root = root.resolve()
    env = dict(os.environ, GIT_OPTIONAL_LOCKS='0')
    try:
        top = subprocess.run(['git', 'rev-parse', '--show-toplevel'], cwd=root,
                             capture_output=True, text=True, timeout=10, env=env)
        if top.returncode or Path(top.stdout.strip()).resolve() != root:
            return unknown
        head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root,
                              capture_output=True, text=True, timeout=10, env=env)
        status = subprocess.run(['git', 'status', '--porcelain=v1', '--untracked-files=all'],
                                cwd=root, capture_output=True, text=True, timeout=10, env=env)
    except (OSError, subprocess.TimeoutExpired):
        return unknown
    if head.returncode or status.returncode:
        return unknown
    modified = bool(status.stdout.strip())
    return dict(git_head=head.stdout.strip(), working_tree_modified=modified,
                git_status='dirty' if modified else 'clean')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(dict(source_manifest(args.root), **git_identification(args.root)), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
