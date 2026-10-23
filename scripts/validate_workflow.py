#!/usr/bin/env python3
"""Exercise engineering invariants in two processes, without altering input images.

Run prepare, then resume with the same fresh output directory. This checks state
integrity and export consistency, not petrographic accuracy. Requires requirements.txt.
"""
import argparse
import ctypes
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from riima import core
from riima import review
from riima.exports import export_batch, staged_file
def memory_mb():
    """Process peak working set on Windows; leave unavailable measurements explicit."""
    if sys.platform != 'win32':
        return None
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize',
            'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
            'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = ctypes.WinDLL('psapi', use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    counter = Counters()
    counter.cb = ctypes.sizeof(counter)
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counter), counter.cb):
        return None
    return round(counter.PeakWorkingSetSize / 1024 ** 2, 1)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def prepare(args):
    if not args.image:
        raise ValueError('Prepare requires at least one --image.')
    args.output.mkdir(parents=True, exist_ok=False)
    hashes = {str(Path(path).resolve()): digest(path) for path in args.image}
    start = time.perf_counter()
    data = core.prepare_analysis(args.image, str(args.output / 'analysis'),
                                 optical_cfg=core.Config(clusters=5, sample_pixels=40000),
                                 mask_mode='overlap_only')
    seconds = time.perf_counter() - start
    interpretation = data['interpretation']
    interpretation.merge_clusters([0, 4])
    interpretation.rename_cluster(0, 'Grupo <validação>')
    interpretation.recolor_cluster(0, (9, 136, 66))
    state = data['review'] = review.ClusterReview.from_prepared(data)
    state.palette[0] = (9, 136, 66)
    state.approve(0)
    item = state.items[0]
    available = item['valid'] & (item['locked'] < 0)
    positives = np.flatnonzero(available & (item['labels'] == 2))[:20]
    negatives = np.flatnonzero(available & (item['labels'] == 1))[:20]
    if not len(positives) or not len(negatives):
        raise ValueError('This engineering example needs populated groups 1 and 2.')
    state.annotate(0, 1, positives, 1)
    state.annotate(0, 1, negatives, -1)
    state.selected_cluster = 1
    state.selected_image = len(data['optical']['image_names']) - 1
    with export_batch():
        np.savez_compressed(staged_file(args.output / 'expected.npz'), initial=item['initial'],
                            locked=item['locked'], labels=item['labels'],
                            positives=positives, negatives=negatives)
    start = time.perf_counter()
    review.save_project(data, args.output / 'pending.riima')
    record = dict(images=hashes, shape=list(item['labels'].shape),
                  feature_dimensions=int(state.centers.shape[1]), preparation_seconds=round(seconds, 3),
                  pending_save_seconds=round(time.perf_counter() - start, 3),
                  prepare_peak_working_set_mb=memory_mb())
    with export_batch():
        staged_file(args.output / 'validation.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps(record, indent=2))


def resume(args):
    import csv
    from riima import report as riima_report
    record = json.loads((args.output / 'validation.json').read_text(encoding='utf-8'))
    start = time.perf_counter()
    data = review.load_project(args.output / 'pending.riima')
    record['resume_seconds'] = round(time.perf_counter() - start, 3)
    state, item = data['review'], data['review'].items[0]
    with np.load(args.output / 'expected.npz', allow_pickle=False) as expected:
        assert state.has_pending and state.completed == {0}
        for name in ('initial', 'labels', 'locked'):
            np.testing.assert_array_equal(item[name], expected[name])
        start = time.perf_counter()
        state.refine_pending(preferred=1)
        record['update_seconds'] = round(time.perf_counter() - start, 3)
        assert not state.has_pending
        assert np.all(item['labels'].ravel()[expected['positives']] == 1)
        assert np.all(item['labels'].ravel()[expected['negatives']] != 1)
        np.testing.assert_array_equal(item['locked'], expected['locked'])
        np.testing.assert_array_equal(item['initial'], expected['initial'])
    state.approve(1)
    protected = item['locked'].copy()
    state.reopen(1)
    np.testing.assert_array_equal(item['locked'][protected == 0], protected[protected == 0])
    review.save_project(data, args.output / 'reviewed.riima')
    restored = review.load_project(args.output / 'reviewed.riima')
    np.testing.assert_array_equal(restored['review'].items[0]['labels'], item['labels'])
    assert restored['review'].models[1].isTrained()
    start = time.perf_counter()
    report = riima_report.export_report(restored, args.output / 'relatorio.html', export_metrics=True)
    record['report_seconds'] = round(time.perf_counter() - start, 3)
    record['resume_peak_working_set_mb'] = memory_mb()
    html = report.read_text(encoding='utf-8')
    assert html.index('id="classificacao"') < html.index('id="tamanho"')
    assert '#098842' in html and 'Grupo &lt;validação&gt;' in html
    with (Path(restored['output']) / 'metrics/group_summary.csv').open(encoding='utf-8-sig', newline='') as file:
        for row in csv.DictReader(file):
            assert int(row['pixels']) == np.count_nonzero(item['labels'] == int(row['group']))
    assert all(digest(path) == original for path, original in record['images'].items())
    record['verified'] = True
    record['report_bytes'] = report.stat().st_size
    with export_batch():
        staged_file(args.output / 'validation.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps(record, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'resume'))
    parser.add_argument('--image', action='append')
    parser.add_argument('--output', type=Path, required=True, help='Fresh engineering-test directory.')
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output.is_relative_to(ROOT):
        parser.error('--output must be outside the source tree.')
    (prepare if args.phase == 'prepare' else resume)(args)


if __name__ == '__main__':
    main()
