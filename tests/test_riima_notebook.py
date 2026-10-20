"""Execute the didactic notebook on tiny inputs without a Jupyter server."""
import json
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from riima import core


def test_notebook_default_inputs_are_generated_in_the_output_directory(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    notebook = json.loads((ROOT / 'notebooks/RIIMA_method_walkthrough.ipynb').read_text(encoding='utf-8'))
    first = next(cell for cell in notebook['cells'] if cell['cell_type'] == 'code')
    namespace = {'RIIMA_OUTPUT_DIR': tmp_path / 'outputs'}
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(plt, 'show', lambda: plt.close('all'))
    try:
        exec(compile(''.join(first['source']), 'notebook-inputs', 'exec'), namespace)
    finally:
        plt.close('all')
    paths = [Path(p).resolve() for p in namespace['paths']]
    assert len(paths) == 2
    assert all(path.is_relative_to(tmp_path) and path.is_file() for path in paths)
    assert all(image.shape == (256, 320, 3) for image in namespace['originals'])


def test_notebook_executes_cluster_workflow_once(tmp_path, monkeypatch):
    nbformat = pytest.importorskip("nbformat")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    notebook = nbformat.read(ROOT / "notebooks/RIIMA_method_walkthrough.ipynb", as_version=4)
    nbformat.validate(notebook)
    headings = [cell.source for cell in notebook.cells if cell.cell_type == "markdown"]
    assert [int(n) for n in re.findall(r"^## (\d+)\.", "\n".join(headings), re.M)] == list(range(1, 14))
    text = "\n".join(cell.source for cell in notebook.cells).lower()
    assert not re.search(r"watershed|grain|markers", text)

    rng = np.random.default_rng(13)
    rgb = rng.integers(55, 220, (64, 72, 3), dtype=np.uint8)
    rgb[12:34, 9:31] = [80, 150, 190]
    rgb[38:58, 40:65] = [180, 165, 80]
    paths = [tmp_path / "synthetic_a.png", tmp_path / "synthetic_b.png"]
    for path in paths:
        Image.fromarray(rgb).save(path)
    output = tmp_path / "exports"
    namespace = {
        "RIIMA_INPUT_PATHS": paths,
        "RIIMA_OUTPUT_DIR": output,
        "RIIMA_CONFIG": core.Config(clusters=4, sample_pixels=1200, kmeans_attempts=1),
        "RIIMA_MASK_MODE": "overlap_only",
    }
    calls = {"register_images": 0, "fit_kmeans": 0}
    for name in calls:
        original = getattr(core, name)

        def counted(*args, _name=name, _original=original, **kwargs):
            calls[_name] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(core, name, counted)
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(plt, "show", lambda: plt.close("all"))
    try:
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type == "code":
                exec(compile(cell.source, f"notebook-cell-{index}", "exec"), namespace)
    finally:
        plt.close("all")

    assert calls == {"register_images": 1, "fit_kmeans": 1}
    state = namespace["state"]
    assert not state.raw["labels"].flags.writeable
    raw = np.load(output / "clusters.npy")
    interpreted = np.load(output / "interpreted_clusters.npy")
    families = np.load(output / "families.npy")
    np.testing.assert_array_equal(raw, namespace["raw_labels"])
    np.testing.assert_array_equal(interpreted, state.interpreted()["labels"])
    np.testing.assert_array_equal(families, state.family_result()["family_map"])
    assert len(state.groups) == 3
    assert state.families[0]["clusters"] == [0, 1, 2]
    assert np.any(families == 1)
    for filename in ["clusters.png", "interpreted_clusters.png", "families.png", "analysis_mask.png",
                     "cluster_statistics.csv", "family_statistics.csv", "interpretation.json"]:
        assert (output / filename).is_file()
    settings = json.loads((output / "analysis_settings.json").read_text(encoding="utf-8"))
    assert settings["feature_space"]["feature_dimension"] == 8
    assert namespace["sample_X"].shape[1] == 8

    # The staged walkthrough must produce the same scientific result as the CLI/GUI core.
    canonical = core.run_analysis([str(p) for p in paths], str(tmp_path / "canonical"),
                                  optical_cfg=namespace["cfg"], mask_mode="overlap_only")
    np.testing.assert_array_equal(raw, canonical["optical"]["labels"])
    np.testing.assert_allclose(namespace["centers"], canonical["optical"]["centers"])
    np.testing.assert_allclose(namespace["confidence"], canonical["optical"]["confidence"])
    np.testing.assert_allclose(namespace["feature_means"], canonical["optical"]["feature_means"], rtol=1e-6)
