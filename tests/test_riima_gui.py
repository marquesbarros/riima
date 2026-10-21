"""Integration tests for the native Tkinter widgets in ``riima.gui``.

``riima_gui.gui()`` accepts an optional ``_test_hook`` that is called with the
function's full local namespace right after the real widget tree is built, instead of
entering ``mainloop``. This drives the actual widgets (real Canvas, real event handler
closures, real ``session`` state dict) rather than mocking the GUI away, without blocking
on an event loop or requiring user interaction.
"""
import time
import dataclasses
import tkinter as tk
from pathlib import Path

import cv2
import numpy as np
import pytest

from riima import core
from riima import gui as riima_gui
from riima import viewer
from riima import review
from test_riima_core_optical import synthetic_registration_image, warp_test_image, write_rgb_images


@pytest.mark.parametrize('geometry', ['1280x820', '1080x700'])
def test_workflow_controls_fit_the_sidebar(app, tmp_path, geometry):
    from tkinter import ttk
    from test_riima_review import prepared_dataset
    app['session']['prepared'] = prepared_dataset(tmp_path, clusters=3)
    app['setup_review']()
    app['refresh_group_listboxes']()
    root = app['root']
    previous = root.geometry()

    def descendants(widget):
        for child in widget.winfo_children():
            yield child
            yield from descendants(child)

    try:
        root.geometry(geometry)
        for step in ('images', 'classify', 'organize', 'refine', 'results'):
            app['go_to_step'](step)
            root.update()
            canvas = app['sidebar_canvas']
            left, right = canvas.winfo_rootx(), canvas.winfo_rootx() + canvas.winfo_width()
            for child in descendants(app['step_frames'][step]):
                if child.winfo_ismapped() and isinstance(child, (
                        ttk.Button, ttk.Checkbutton, ttk.Radiobutton, ttk.Combobox, tk.Listbox)):
                    x = child.winfo_rootx()
                    assert left <= x and x + child.winfo_width() <= right + 1, (step, child)
            assert app['preview_canvas'].winfo_width() > 300
    finally:
        root.geometry(previous)


@pytest.mark.parametrize('failure', ['missing_image', 'invalid_k', 'overlap'])
def test_failed_reclassification_keeps_the_current_review_and_workspace(app, tmp_path, monkeypatch, failure):
    from test_riima_review import prepared_dataset
    prepared = prepared_dataset(tmp_path, count=1)
    state = prepared['review']
    state.approve(1)
    pixel = int(np.flatnonzero(state.items[0]['labels'] == 0)[0])
    state.annotate(0, 0, [pixel], 1)
    session = app['session']
    session['prepared'] = prepared
    session['project_path'] = str(tmp_path / 'current.riima')
    session['images'] = [dict(path=state.items[0]['path'], name=state.items[0]['name'])]
    if failure == 'missing_image':
        session['images'][0]['path'] = str(tmp_path / 'missing.png')
    app['out_var'].set(prepared['output'])
    app['clusters_var'].set(2)
    app['reference_var'].set(0)
    app['rebuild_image_rows']()
    app['setup_review']()
    app['go_to_step']('classify')
    app['view_type_var'].set('compare')
    app['on_view_type_change']()
    app['refresh_preview_content']()
    before = session['current_rgb'].copy()
    viewport = session['viewport']
    approved = state.items[0]['locked'].copy()
    messages = []
    monkeypatch.setattr(app['messagebox'], 'showerror', lambda *args: messages.append(args))
    if failure == 'invalid_k':
        app['clusters_var'].set('invalid')
    elif failure == 'overlap':
        def insufficient(*args, **kwargs):
            raise core.InsufficientOverlapError('Not enough overlap for this input')
        monkeypatch.setattr(core, 'prepare_analysis', insufficient)
    app['analyze']()
    assert messages
    assert session['prepared'] is prepared
    assert session['viewport'] is viewport
    assert app['view_type_var'].get() == 'compare'
    assert session['compare_mode']
    assert app['workflow_var'].get() == 'classify'
    np.testing.assert_array_equal(session['current_rgb'], before)
    np.testing.assert_array_equal(state.items[0]['locked'], approved)
    assert state.has_pending and state.items[0]['annotations'][0].flat[pixel] == 1
    restored = review.load_project(session['project_path'])
    np.testing.assert_array_equal(restored['review'].items[0]['locked'], approved)
    assert restored['review'].has_pending
    assert str(app['analyze_btn']['state']) == 'normal'


def test_results_report_button_exports_html_and_cancellation_does_not_compute(app, tmp_path, monkeypatch):
    from test_riima_review import prepared_dataset
    from riima import metrics as riima_metrics
    prepared = prepared_dataset(tmp_path, count=1)
    app['session']['prepared'] = prepared
    app['setup_review']()
    app['go_to_step']('results')
    buttons = [w for w in app['step_frames']['results'].winfo_children() if _text_of(w) == 'Exportar relatório']
    assert len(buttons) == 1
    target = tmp_path / 'report.html'
    with monkeypatch.context() as cancel:
        cancel.setattr(app['filedialog'], 'asksaveasfilename', lambda **kw: '')
        cancel.setattr(riima_metrics, 'export_metrics', lambda *a, **kw: pytest.fail('calculation after cancellation'))
        buttons[0].invoke()
    assert not target.exists()
    monkeypatch.setattr(app['filedialog'], 'asksaveasfilename', lambda **kw: str(target))
    monkeypatch.setattr(app['messagebox'], 'showerror', lambda title, text: pytest.fail(text))
    buttons[0].invoke()
    html = target.read_text(encoding='utf-8')
    assert html.index('id="classificacao"') < html.index('id="tamanho"')
    assert app['workflow_var'].get() == 'results'
    assert app['session']['prepared'] is prepared


def test_results_report_blocks_pending_marks_without_applying_them(app, tmp_path, monkeypatch):
    from test_riima_review import prepared_dataset
    prepared = prepared_dataset(tmp_path, count=1)
    app['session']['prepared'] = prepared
    app['setup_review']()
    before = prepared['review'].items[0]['labels'].copy()
    prepared['review'].annotate(0, 0, [212], 1)
    messages = []
    monkeypatch.setattr(app['filedialog'], 'asksaveasfilename', lambda **kw: pytest.fail('picker before pending check'))
    monkeypatch.setattr(app['messagebox'], 'showinfo', lambda title, text: messages.append(text))
    app['export_descriptive_report']()
    assert messages and 'Atualiz' in messages[0]
    assert prepared['review'].has_pending
    np.testing.assert_array_equal(prepared['review'].items[0]['labels'], before)


@pytest.mark.parametrize('exporter', ['export_current', 'export_selected_mask'])
def test_png_export_failure_preserves_previous_file_and_current_analysis(app, tmp_path, monkeypatch, exporter):
    from PIL import Image
    from test_riima_review import prepared_dataset
    prepared = prepared_dataset(tmp_path, count=1)
    app['session']['prepared'] = prepared
    app['setup_review']()
    app['go_to_step']('organize')
    app['view_type_var'].set('layers')
    app['on_view_type_change']()
    app['session']['workspace_selected_groups'] = [0]
    app['refresh_preview_content']()
    labels = prepared['review'].items[0]['labels'].copy()
    target = tmp_path / 'existing.png'
    Image.new('RGB', (8, 8), '#098842').save(target)
    previous = target.read_bytes()
    monkeypatch.setattr(app['filedialog'], 'asksaveasfilename', lambda **kw: str(target))
    messages = []
    monkeypatch.setattr(app['messagebox'], 'showerror', lambda *args: messages.append(args))
    def interrupted_save(image, file, *args, **kwargs):
        Path(file).write_bytes(b'incomplete PNG stream')
        raise OSError('Disk full during PNG export')
    monkeypatch.setattr(Image.Image, 'save', interrupted_save)
    app[exporter]()
    assert messages and 'Disk full' in messages[0][1]
    assert target.read_bytes() == previous
    assert app['session']['prepared'] is prepared
    np.testing.assert_array_equal(prepared['review'].items[0]['labels'], labels)
    assert not list(tmp_path.glob('.riima-png-*'))


def test_click_replaces_group_selection_and_shift_adds_individual_groups(app, tmp_path):
    from test_riima_review import prepared_dataset
    prepared = prepared_dataset(tmp_path, clusters=3)
    app['session']['prepared'] = prepared
    app['setup_review']()
    app['go_to_step']('organize')
    app['refresh_group_listboxes']()
    app['root'].update_idletasks()
    groups = app['optical_group_list']
    before = [item['labels'].copy() for item in prepared['review'].items]
    def click(index, state=0):
        x, y, width, height = groups.bbox(index)
        groups.event_generate('<ButtonPress-1>', x=x + 4, y=y + height // 2, state=state)
        groups.event_generate('<ButtonRelease-1>', x=x + 4, y=y + height // 2, state=state)
        app['root'].update()
    click(0)
    click(2)
    assert groups.curselection() == (2,)
    click(0, 1)  # Shift adds a nonadjacent group without selecting the intervening one.
    assert groups.curselection() == (0, 2)
    assert app['selected_workspace_groups']() == [0, 2]
    click(0, 1)
    assert groups.curselection() == (2,)
    click(1, 4)  # Ctrl without Shift still replaces the selection.
    assert groups.curselection() == (1,)
    for item, labels in zip(prepared['review'].items, before):
        np.testing.assert_array_equal(item['labels'], labels)


def test_complete_registered_workflow_reviews_resumes_and_exports_consistent_results(app, tmp_path, monkeypatch):
    import base64
    import csv
    import io
    from types import SimpleNamespace
    from PIL import Image
    from test_riima_report import ImagesInReport

    directory = tmp_path / 'análise com acentos'
    directory.mkdir()
    reference = synthetic_registration_image()
    paths = []
    for i, rgb in enumerate((reference, warp_test_image(reference, dx=3, dy=-2))):
        path = directory / f'lâmina_ç_{i}.png'
        Image.fromarray(rgb).save(path)
        paths.append(str(path))
    originals = [Path(path).read_bytes() for path in paths]
    monkeypatch.setattr(app['filedialog'], 'askopenfilenames', lambda **kw: paths)
    monkeypatch.setattr(app['messagebox'], 'showerror', lambda title, text: pytest.fail(text))
    app['automatic_output_var'].set(True)
    app['choose_images']()
    app['next_btn'].invoke()
    app['clusters_var'].set(4)
    app['mask_mode_var'].set('Somente sobreposição comum')
    app['analyze_btn'].invoke()
    prepared = app['session']['prepared']
    assert prepared['optical']['sample_X'].shape[1] == 8
    assert app['workflow_var'].get() == 'classify'
    app['next_btn'].invoke()
    groups = app['optical_group_list']
    groups.selection_clear(0, 'end')
    groups.selection_set(0)
    groups.selection_set(3)
    app['on_group_selection']()
    app['merge_optical_selected']()
    state = prepared['review']
    assert state.groups == {0: {0, 3}, 1: {1}, 2: {2}}
    groups.selection_clear(0, 'end')
    groups.selection_set(0)
    app['on_group_selection']()
    monkeypatch.setattr(app['simpledialog'], 'askstring', lambda *args, **kw: 'Grupo <á & ç>')
    app['rename_selected_group'](groups, 'cluster')
    monkeypatch.setattr(app['colorchooser'], 'askcolor', lambda **kw: ((9, 136, 66), '#098842'))
    app['recolor_selected_group'](groups, 'cluster')
    app['next_btn'].invoke()
    app['review_cluster_combo'].current(state.class_ids.index(0))
    app['select_review_cluster']()
    app['review_approve_btn'].invoke()
    item = state.items[0]
    raw = item['initial'].copy()
    frozen = item['locked'].copy()
    app['review_cluster_combo'].current(state.class_ids.index(1))
    app['select_review_cluster']()
    app['review_display_var'].set('Máscara selecionada')
    app['change_preview']()
    app['review_radius_var'].set(1)

    def paint(tool, cluster):
        candidates = (item['valid'] & (item['labels'] == cluster) & (item['locked'] < 0)).astype(np.uint8)
        interior = cv2.erode(candidates, np.ones((5, 5), np.uint8))
        choices = np.flatnonzero(interior if np.any(interior) else candidates)
        pixel = int(choices[len(choices) // 2])
        y, x = divmod(pixel, item['labels'].shape[1])
        canvas = app['preview_canvas']
        point = viewer.image_to_canvas((x + .5, y + .5), app['session']['viewport'],
                                       item['labels'].shape, (canvas.winfo_width(), canvas.winfo_height()))
        app['review_tool_buttons'][tool].invoke()
        event = SimpleNamespace(x=point[0], y=point[1])
        app['on_canvas_press'](event)
        app['on_canvas_release'](event)
        return pixel

    positive = paint('positive', 2)
    negative = paint('negative', 1)
    assert state.has_pending
    project = directory / 'projeto retomável.riima'
    monkeypatch.setattr(app['filedialog'], 'asksaveasfilename', lambda **kw: str(project))
    assert app['save_project_current'](choose=True)
    app['session']['prepared'] = None
    monkeypatch.setattr(app['filedialog'], 'askopenfilename', lambda **kw: str(project))
    app['open_project']()
    prepared = app['session']['prepared']
    state = prepared['review']
    item = state.items[0]
    assert app['review_display_var'].get() == 'Máscara selecionada'
    assert state.has_pending and state.completed == {0}
    np.testing.assert_array_equal(item['locked'], frozen)
    app['review_update_btn'].invoke()
    assert not state.has_pending
    assert item['labels'].flat[positive] == 1 and item['labels'].flat[negative] != 1
    np.testing.assert_array_equal(item['locked'], frozen)
    np.testing.assert_array_equal(item['initial'], raw)
    app['review_approve_btn'].invoke()
    other = item['locked'] == 0
    app['review_reopen_btn'].invoke()
    assert state.completed == {0}
    np.testing.assert_array_equal(item['locked'][other], frozen[other])
    app['review_image_combo'].current(1)
    app['select_review_image']()
    app['view_type_var'].set('compare')
    app['on_view_type_change']()
    assert app['compare_panel'].winfo_manager() and not app['display_controls'].winfo_manager()
    assert app['save_project_current']()
    app['go_to_step']('results')
    mask_file = directory / 'máscara_ç.png'
    monkeypatch.setattr(app['filedialog'], 'asksaveasfilename', lambda **kw: str(mask_file))
    app['export_selected_mask']()
    np.testing.assert_array_equal(np.asarray(Image.open(mask_file)),
                                  viewer.render_workspace(prepared, 'Máscara selecionada', selected_groups=[1]))
    report_file = directory / 'relatório_á.html'
    monkeypatch.setattr(app['filedialog'], 'asksaveasfilename', lambda **kw: str(report_file))
    app['export_descriptive_report']()
    html = report_file.read_text(encoding='utf-8')
    assert '#098842' in html and 'Grupo &lt;á &amp; ç&gt;' in html
    assert html.index('id="classificacao"') < html.index('id="tamanho"')
    parser = ImagesInReport()
    parser.feed(html)
    final = next(img for img in parser.images if img.get('class') == 'classification-map')
    actual = np.asarray(Image.open(io.BytesIO(base64.b64decode(final['src'].split(',', 1)[1]))))
    np.testing.assert_array_equal(actual, viewer.render_workspace(prepared, 'Classificação'))
    with (Path(prepared['output']) / 'metrics' / 'group_summary.csv').open(encoding='utf-8-sig', newline='') as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        assert int(row['pixels']) == np.count_nonzero(item['labels'] == int(row['group']))
    assert [Path(path).read_bytes() for path in paths] == originals


@pytest.mark.parametrize('image_count', [1, 2])
def test_classification_stays_in_current_stage_for_k_evaluation(app, tmp_path, monkeypatch, image_count):
    rgb = synthetic_registration_image()
    paths = write_rgb_images(tmp_path, [rgb, warp_test_image(rgb, dx=3, dy=-2)][:image_count])
    app['session']['prepared'] = None
    app['session']['images'] = [dict(path=path, name=Path(path).name) for path in paths]
    app['rebuild_image_rows']()
    app['reference_var'].set(0)
    app['mask_mode_var'].set('Somente sobreposição comum')
    app['clusters_var'].set(2)
    app['out_var'].set(str(tmp_path / 'new-analysis'))
    app['go_to_step']('classify')
    monkeypatch.setattr(app['messagebox'], 'showerror', lambda title, text: pytest.fail(text))
    app['analyze']()
    assert app['workflow_var'].get() == 'classify'
    assert app['step_frames']['classify'].winfo_manager() == 'pack'
    assert app['step_frames']['organize'].winfo_manager() == ''
    prepared = app['session']['prepared']
    assert prepared['optical']['sample_X'].shape[1] == 4 * image_count
    assert prepared['review'].mode == 'registered'
    assert len(prepared['review'].items) == 1
    assert len(prepared['optical']['registered_images']) == image_count


def test_new_images_save_current_analysis_and_start_with_separate_output(app, tmp_path, monkeypatch):
    from test_riima_review import prepared_dataset
    old = prepared_dataset(tmp_path)
    old['review'].approve(1)
    annotated_pixel = int(np.flatnonzero(old['review'].items[0]['labels'] == 0)[0])
    old['review'].annotate(0, 0, [annotated_pixel], 1)
    approved_masks = [item['locked'].copy() for item in old['review'].items]
    old_project = tmp_path / 'previous.riima'
    app['session']['prepared'] = old
    app['session']['project_path'] = str(old_project)
    app['out_var'].set(old['output'])
    app['session']['images'] = [dict(path=item['path'], name=item['name']) for item in old['review'].items]
    app['setup_review']()
    new_folder = tmp_path / 'another-field'
    new_folder.mkdir()
    (new_folder / 'riima_output').mkdir()  # Existing results must not be overwritten.
    new_image = new_folder / 'slide.png'
    new_image.write_bytes(b'picker does not decode input yet')
    monkeypatch.setattr(app['filedialog'], 'askopenfilenames', lambda **kw: (str(new_image),))
    app['automatic_output_var'].set(True)
    app['choose_images']()
    restored = review.load_project(old_project)
    assert restored['review'].items[0]['annotations'][0].flat[annotated_pixel] == 1
    assert restored['review'].completed == {1}
    for item, approved in zip(restored['review'].items, approved_masks):
        np.testing.assert_array_equal(item['locked'], approved)
    assert restored['output_history'] == [old['output']]
    assert restored['output'] == str(tmp_path / 'previous_output')
    assert not Path(restored['output']).exists()
    assert not Path(old['output']).exists()
    assert app['session']['prepared'] is None
    assert app['session']['project_path'] is None
    assert app['session']['current_rgb'] is None
    assert Path(app['out_var'].get()).parent == new_folder
    assert Path(app['out_var'].get()).name == 'riima_output_2'
    assert app['session']['images'][0]['path'] == str(new_image)


def test_new_image_picker_cancel_and_save_failure_preserve_current_analysis(app, tmp_path, monkeypatch):
    from test_riima_review import prepared_dataset
    old = prepared_dataset(tmp_path)
    app['session']['prepared'] = old
    old_images = [dict(path=item['path'], name=item['name']) for item in old['review'].items]
    app['session']['images'] = old_images
    app['out_var'].set(old['output'])
    monkeypatch.setattr(app['filedialog'], 'askopenfilenames', lambda **kw: ())
    app['choose_images']()
    assert app['session']['prepared'] is old
    assert app['session']['images'] == old_images
    monkeypatch.setattr(app['filedialog'], 'askopenfilenames', lambda **kw: (str(tmp_path / 'new.png'),))
    monkeypatch.setattr(review, 'save_project', lambda *args: (_ for _ in ()).throw(OSError('save failed')))
    monkeypatch.setattr(app['messagebox'], 'showerror', lambda *args: None)
    app['choose_images']()
    assert app['session']['prepared'] is old
    assert app['session']['images'] == old_images
    assert app['out_var'].get() == old['output']


def test_manual_output_remains_selected_when_automatic_folder_is_disabled(app, tmp_path, monkeypatch):
    app['session']['prepared'] = None
    app['out_var'].set(str(tmp_path / 'chosen-output'))
    app['automatic_output_var'].set(False)
    monkeypatch.setattr(app['filedialog'], 'askopenfilenames', lambda **kw: (str(tmp_path / 'different.png'),))
    app['choose_images']()
    assert Path(app['out_var'].get()) == tmp_path / 'chosen-output'
    app['automatic_output_var'].set(True)


def test_mask_brush_retains_display_and_undo_restores_the_stroke(app, tmp_path):
    from test_riima_review import prepared_dataset
    from types import SimpleNamespace
    prepared = prepared_dataset(tmp_path)
    app['session']['prepared'] = prepared
    app['setup_review']()
    app['go_to_step']('refine')
    app['review_radius_var'].set(1)
    app['review_display_var'].set('Máscara selecionada')
    app['review_tool_var'].set('negative')
    app['activate_review_tool']()
    canvas = app['preview_canvas']
    shape = prepared['review'].items[0]['valid'].shape
    x, y = viewer.image_to_canvas((3, 8), app['session']['viewport'], shape,
                                  (canvas.winfo_width(), canvas.winfo_height()))
    event = SimpleNamespace(x=int(x), y=int(y))
    app['on_canvas_press'](event)
    app['on_canvas_release'](event)
    assert app['review_display_var'].get() == 'Máscara selecionada'
    assert np.any(prepared['review'].items[0]['annotations'][0] == -1)
    app['undo_review_stroke']()
    assert not np.any(prepared['review'].items[0]['annotations'][0])
    assert not prepared['review'].has_pending


def test_sidebar_wheel_does_not_move_short_content(app):
    from types import SimpleNamespace
    app['go_to_step']('images')
    app['root'].update_idletasks()
    canvas = app['sidebar_canvas']
    canvas.configure(scrollregion=(0, 0, canvas.winfo_width(), canvas.winfo_height() - 10))
    for _ in range(6):
        assert app['_sidebar_wheel'](SimpleNamespace(delta=-120)) == 'break'
    assert canvas.yview()[0] == 0


def test_sidebar_wheel_clamps_both_ends_of_long_content(app):
    from types import SimpleNamespace
    canvas = app['sidebar_canvas']
    height = canvas.winfo_height()
    canvas.configure(scrollregion=(0, 0, canvas.winfo_width(), height * 2))
    for _ in range(80):
        app['_sidebar_wheel'](SimpleNamespace(delta=-120))
    assert 0 <= canvas.yview()[0] <= .5
    assert canvas.yview()[1] == 1
    for _ in range(80):
        app['_sidebar_wheel'](SimpleNamespace(delta=120))
    assert canvas.yview()[0] == 0


def test_clear_selection_removes_mask_without_modifying_classification(app, tmp_path):
    from test_riima_review import prepared_dataset
    prepared = prepared_dataset(tmp_path)
    app['session']['prepared'] = prepared
    app['setup_review']()
    app['go_to_step']('organize')
    app['refresh_group_listboxes']()
    app['optical_group_list'].selection_set(0, 1)
    app['on_group_selection']()
    app['review_display_var'].set('Máscara selecionada')
    before = [item['labels'].copy() for item in prepared['review'].items]
    app['clear_group_selection']()
    assert app['selected_workspace_groups']() == []
    assert app['optical_group_list'].curselection() == ()
    assert not np.any(app['session']['current_rgb'])
    for item, labels in zip(prepared['review'].items, before):
        np.testing.assert_array_equal(item['labels'], labels)


def test_compare_hides_work_controls_and_return_restores_them(app, tmp_path):
    from test_riima_review import prepared_dataset
    app['session']['prepared'] = prepared_dataset(tmp_path)
    app['setup_review']()
    app['view_type_var'].set('compare')
    app['on_view_type_change']()
    assert app['display_controls'].winfo_manager() == ''
    assert app['compare_panel'].winfo_manager() == 'pack'
    app['view_type_var'].set('layers')
    app['on_view_type_change']()
    assert app['display_controls'].winfo_manager() == 'pack'
    assert app['compare_panel'].winfo_manager() == ''


def test_restore_tool_reactivates_excluded_area_without_erasing_other_examples(app, tmp_path):
    from test_riima_review import prepared_dataset
    from types import SimpleNamespace
    prepared = prepared_dataset(tmp_path)
    app['session']['prepared'] = prepared
    app['setup_review']()
    state = prepared['review']
    state.exclude(0, [195], True)
    state.refine_pending()
    state.annotate(0, 0, [212], 1)
    app['go_to_step']('refine')
    app['review_radius_var'].set(1)
    app['review_tool_var'].set('restore')
    app['activate_review_tool']()
    canvas = app['preview_canvas']
    shape = state.items[0]['valid'].shape
    x, y = viewer.image_to_canvas((3, 8), app['session']['viewport'], shape,
                                  (canvas.winfo_width(), canvas.winfo_height()))
    event = SimpleNamespace(x=int(x), y=int(y))
    app['on_canvas_press'](event)
    app['on_canvas_release'](event)
    assert not state.items[0]['excluded'].flat[195]
    assert state.items[0]['annotations'][0].flat[212] == 1
    app['refine_review_cluster']()
    assert state.items[0]['labels'].flat[195] != -1


def test_review_gui_stroke_approval_navigation_and_project_resume(app, tmp_path, monkeypatch):
    from test_riima_review import prepared_dataset
    from types import SimpleNamespace
    prepared = prepared_dataset(tmp_path)
    app['session']['prepared'] = prepared
    app['session']['project_path'] = str(tmp_path / 'gui.riima')
    app['setup_review']()
    app['review_enabled_var'].set(True)
    app['review_tool_var'].set('positive')
    app['review_cluster_combo'].current(0)
    app['select_review_cluster']()
    app['refresh_preview_content']()
    canvas = app['preview_canvas']
    shape = prepared['review'].items[0]['valid'].shape
    viewport_size = (canvas.winfo_width(), canvas.winfo_height())
    x, y = viewer.image_to_canvas((3, 8), app['session']['viewport'], shape, viewport_size)
    event = SimpleNamespace(x=int(x), y=int(y))
    app['on_canvas_press'](event)
    app['on_canvas_release'](event)
    assert np.any(prepared['review'].items[0]['annotations'][0] == 1)
    app['refine_review_cluster']()
    app['approve_review_cluster']()
    assert 0 in prepared['review'].completed
    app['review_image_combo'].current(1)
    app['select_review_image']()
    assert prepared['review'].selected_image == 1
    assert app['session']['current_image_shape'] == (20, 28)
    app['save_project_current']()
    monkeypatch.setattr(app['filedialog'], 'askopenfilename', lambda **kwargs: str(tmp_path / 'gui.riima'))
    app['open_project']()
    restored = app['session']['prepared']['review']
    assert restored.completed == {0}
    assert restored.selected_image == 1
    app['review_cluster_combo'].current(0)
    app['select_review_cluster']()
    app['reopen_review_cluster']()
    assert restored.completed == set()
    app['review_enabled_var'].set(False)
    app['review_tool_var'].set('pan')


def test_manual_review_update_cancels_pending_timer(app, tmp_path):
    from test_riima_review import prepared_dataset
    app['session']['prepared'] = prepared_dataset(tmp_path)
    app['setup_review']()
    timer = app['root'].after(1000, app['refine_review_cluster'])
    app['session']['review_after'] = timer
    try:
        app['refine_review_cluster']()
        assert timer not in app['root'].tk.call('after', 'info')
    finally:
        app['root'].after_cancel(timer)


def test_joint_analysis_requires_active_reference_after_loading_old_project(app, tmp_path, monkeypatch):
    from test_riima_review import prepared_dataset
    seed = prepared_dataset(tmp_path)
    archived = tmp_path / 'legacy.riima'
    review.save_project(seed, archived)
    app['session']['prepared'] = None
    monkeypatch.setattr(app['filedialog'], 'askopenfilename', lambda **kw: str(archived))
    app['open_project']()
    loaded = app['session']['prepared']
    assert loaded['review'].mode == 'independent'
    app['session']['_active_vars'][0].set(False)
    app['reference_var'].set(0)
    warnings = []
    monkeypatch.setattr(app['messagebox'], 'showwarning', lambda title, text: warnings.append(text))
    monkeypatch.setattr(app['messagebox'], 'showerror', lambda title, text: pytest.fail(text))
    app['analyze']()
    assert warnings and 'referência' in warnings[0]
    assert app['session']['prepared'] is loaded
    for item, original in zip(loaded['review'].items, seed['review'].items):
        np.testing.assert_array_equal(item['initial'], original['initial'])


def test_marking_and_navigation_only_accumulate_edits(app, tmp_path, monkeypatch):
    from test_riima_review import prepared_dataset
    from types import SimpleNamespace
    prepared = prepared_dataset(tmp_path)
    app['session']['prepared'] = prepared
    app['setup_review']()
    app['review_enabled_var'].set(True)
    app['review_tool_var'].set('positive')
    app['review_display_var'].set('Sobreposição')
    app['isolate_var'].set(True)
    app['refresh_preview_content']()
    state = prepared['review']
    original_images = [item['rgb'].copy() for item in state.items]
    original_overlay = viewer.render_workspace
    monkeypatch.setattr(viewer, 'render_workspace', lambda *args, **kwargs: pytest.fail('full overlay per stroke'))
    before = [item['labels'].copy() for item in state.items]
    monkeypatch.setattr(state, 'refine', lambda *args: pytest.fail('automatic refinement'))
    monkeypatch.setattr(state, 'refine_pending', lambda *args, **kwargs: pytest.fail('automatic batch refinement'))
    monkeypatch.setattr(review, 'save_project', lambda *args: pytest.fail('automatic full save'))
    for timer in ('review_after', 'autosave_after'):
        if app['session'].get(timer) is not None:
            app['root'].after_cancel(app['session'][timer])
            app['session'][timer] = None
    canvas = app['preview_canvas']
    shape = state.items[0]['valid'].shape
    x, y = viewer.image_to_canvas((3, 8), app['session']['viewport'], shape,
                                  (canvas.winfo_width(), canvas.winfo_height()))
    event = SimpleNamespace(x=int(x), y=int(y))
    app['on_canvas_press'](event)
    app['on_canvas_release'](event)
    assert state.dirty == {0}
    assert app['session'].get('review_after') is None
    assert app['session'].get('autosave_after') is None
    monkeypatch.setattr(viewer, 'render_workspace', original_overlay)
    app['review_cluster_combo'].current(1)
    app['select_review_cluster']()
    app['review_image_combo'].current(1)
    app['select_review_image']()
    assert app['session'].get('autosave_after') is None
    for item, labels in zip(state.items, before):
        np.testing.assert_array_equal(item['labels'], labels)
    for item, rgb in zip(state.items, original_images):
        np.testing.assert_array_equal(item['rgb'], rgb)
    app['review_enabled_var'].set(False)
    app['review_tool_var'].set('pan')


def test_gui_refines_named_groups_with_non_contiguous_ids_and_safe_regrouping(app, tmp_path, monkeypatch):
    from test_riima_review import prepared_dataset
    prepared = prepared_dataset(tmp_path, clusters=3)
    app['session']['prepared'] = prepared
    app['setup_review']()
    app['refresh_group_listboxes']()
    app['optical_group_list'].selection_set(0, 1)
    app['merge_optical_selected']()
    state = prepared['review']
    assert state.groups == {0: {0, 1}, 2: {2}}
    app['apply_interpretation']('rename_cluster', 0, 'Matriz')
    assert 'Matriz' in app['review_cluster_combo']['values'][0]
    app['review_cluster_combo'].current(1)
    app['select_review_cluster']()
    assert state.selected_cluster == 2
    state.annotate(0, 2, [212], 1)
    warnings = []
    monkeypatch.setattr(app['messagebox'], 'showinfo', lambda title, message: warnings.append(message))
    app['optical_group_list'].selection_set(0, 1)
    app['merge_optical_selected']()
    assert warnings
    assert state.groups == {0: {0, 1}, 2: {2}}
    assert prepared['interpretation'].groups == [{0, 1}, {2}]
    assert state.items[0]['annotations'][2].flat[212] == 1


def test_legacy_review_can_edit_interpretation_metadata_without_regrouping(app, tmp_path, monkeypatch):
    import json
    from test_riima_review import prepared_dataset
    prepared = prepared_dataset(tmp_path)
    prepared['interpretation'].merge_clusters([0, 1])
    prepared['review'].annotate(0, 0, [212], 1)
    path = tmp_path / 'legacy.riima'
    review.save_project(prepared, path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    metadata = json.loads(str(arrays['metadata']))
    metadata['version'] = 1
    for i in range(1, len(metadata['items'])):
        arrays[f'{i}_sample_X'] = arrays['0_sample_X'].copy()
    metadata.pop('groups')
    arrays['metadata'] = np.array(json.dumps(metadata))
    with path.open('wb') as handle:
        np.savez_compressed(handle, **arrays)
    loaded = review.load_project(path)
    app['session']['prepared'] = loaded
    app['setup_review']()
    monkeypatch.setattr(app['messagebox'], 'showinfo', lambda *args: pytest.fail(str(args)))
    app['apply_interpretation']('rename_cluster', 0, 'Matriz')
    app['apply_interpretation']('create_family', [0], 'Silicatos')
    app['apply_interpretation']('undo')
    assert loaded['interpretation'].names == {0: 'Matriz'}
    assert not loaded['interpretation'].families
    assert loaded['review'].groups == {0: {0}, 1: {1}}
    assert loaded['review'].items[0]['annotations'][0].flat[212] == 1


def test_guided_workspace_isolates_groups_without_separate_family_or_layer_panels(app, tmp_path):
    from test_riima_review import prepared_dataset
    prepared = prepared_dataset(tmp_path, clusters=3)
    app['session']['prepared'] = prepared
    app['setup_review']()
    app['refresh_group_listboxes']()
    app['go_to_step']('organize')
    assert app['step_frames']['organize'].winfo_manager() == 'pack'
    assert app['step_frames']['refine'].winfo_manager() == ''
    assert app['family_group_frame'].winfo_manager() == ''
    assert app['layers_panel'].winfo_manager() == ''
    app['optical_group_list'].selection_clear(0, 'end')
    app['optical_group_list'].selection_set(0, 1)
    app['on_group_selection']()
    app['review_display_var'].set('Máscara selecionada')
    app['change_preview']()
    expected = viewer.workspace_mask(prepared, [0, 1])
    np.testing.assert_array_equal(app['session']['current_rgb'][..., 0], expected.astype(np.uint8) * 255)
    app['merge_optical_selected']()
    assert prepared['review'].groups == {0: {0, 1}, 2: {2}}
    app['go_to_step']('refine')
    assert prepared['review'].selected_cluster == 0
    assert set(app['session']['_compare_sources']) == {'Imagem original', 'Classificação atual', 'Máscara selecionada'}
    app['go_to_step']('results')
    assert app['step_frames']['results'].winfo_manager() == 'pack'
    assert app['step_frames']['organize'].winfo_manager() == ''


def test_workspace_resume_restores_step_image_and_mask_selection(app, tmp_path, monkeypatch):
    from test_riima_review import prepared_dataset
    prepared = prepared_dataset(tmp_path)
    app['session']['prepared'] = prepared
    app['session']['project_path'] = str(tmp_path / 'workspace.riima')
    app['setup_review']()
    app['refresh_group_listboxes']()
    app['go_to_step']('organize')
    app['optical_group_list'].selection_clear(0, 'end')
    app['optical_group_list'].selection_set(1)
    app['on_group_selection']()
    app['review_image_combo'].current(1)
    app['select_review_image']()
    app['review_display_var'].set('Máscara selecionada')
    app['change_preview']()
    app['automatic_output_var'].set(False)
    app['save_project_current']()
    monkeypatch.setattr(app['filedialog'], 'askopenfilename', lambda **kw: str(tmp_path / 'workspace.riima'))
    app['session']['prepared'] = None
    app['session']['workspace_selected_groups'] = []
    app['automatic_output_var'].set(True)
    app['workflow_var'].set('images')
    app['review_display_var'].set('Imagem')
    app['open_project']()
    assert app['workflow_var'].get() == 'organize'
    assert app['review_display_var'].get() == 'Máscara selecionada'
    assert app['session']['prepared']['review'].selected_image == 1
    assert app['session']['workspace_selected_groups'] == [1]
    assert app['automatic_output_var'].get() is False
    app['automatic_output_var'].set(True)


def test_legacy_saved_association_is_a_current_mask_inside_organize(app, tmp_path):
    from test_riima_review import prepared_dataset
    prepared = prepared_dataset(tmp_path, clusters=3)
    prepared['interpretation'].create_family([0, 1], 'Matriz')
    state = prepared['review']
    state.annotate(0, 0, [212], 1)
    state.annotate(0, 0, [195], -1)
    state.refine_pending()
    app['session']['prepared'] = prepared
    app['setup_review']()
    app['refresh_group_listboxes']()
    app['go_to_step']('organize')
    app['family_group_list'].selection_set(0)
    app['select_legacy_mask']()
    assert app['session']['workspace_selected_groups'] == [0, 1]
    assert app['review_display_var'].get() == 'Máscara selecionada'
    expected = viewer.workspace_mask(prepared, [0, 1])
    np.testing.assert_array_equal(app['session']['current_rgb'][..., 0], expected.astype(np.uint8) * 255)
    assert 'Famílias' not in app['session']['_compare_sources']


def test_stage_navigation_preserves_pending_edits_without_training(app, tmp_path, monkeypatch):
    from test_riima_review import prepared_dataset
    app['session']['prepared'] = prepared_dataset(tmp_path)
    app['setup_review']()
    state = app['session']['prepared']['review']
    state.annotate(0, 0, [212], 1)
    before = [item['labels'].copy() for item in state.items]
    monkeypatch.setattr(state, 'refine_pending', lambda **kw: pytest.fail('training while navigating'))
    for step in ('organize', 'refine', 'results', 'organize'):
        app['go_to_step'](step)
        assert state.dirty == {0}
        assert [key for key, frame in app['step_frames'].items() if frame.winfo_manager()] == [step]
    for item, initial in zip(state.items, before):
        np.testing.assert_array_equal(item['labels'], initial)


def test_comparator_resume_restores_union_mask_and_presentation_settings(app, tmp_path, monkeypatch):
    from test_riima_review import prepared_dataset
    prepared = prepared_dataset(tmp_path)
    app['session']['prepared'] = prepared
    app['setup_review']()
    app['refresh_group_listboxes']()
    app['go_to_step']('organize')
    app['optical_group_list'].selection_set(0, 1)
    app['on_group_selection']()
    app['review_opacity_var'].set(30)
    app['view_type_var'].set('compare')
    app['on_view_type_change']()
    app['split_right_var'].set('Máscara selecionada')
    app['on_compare_source_change']()
    archived = tmp_path / 'organize.riima'
    review.save_project(prepared, archived)
    app['session']['project_path'] = str(tmp_path / 'other.riima')
    app['go_to_step']('refine')
    app['review_opacity_var'].set(85)
    monkeypatch.setattr(app['filedialog'], 'askopenfilename', lambda **kw: str(archived))
    app['open_project']()
    restored = app['session']['prepared']
    expected = viewer.workspace_mask(restored, [0, 1])
    np.testing.assert_array_equal(app['session']['compare_cache']['right_rgb'][..., 0], expected.astype(np.uint8) * 255)
    assert app['review_opacity_var'].get() == 30
    assert app['view_type_var'].get() == 'compare'


def test_deselecting_groups_clears_mask_and_refine_selection_follows_to_results(app, tmp_path):
    from test_riima_review import prepared_dataset
    app['session']['prepared'] = prepared_dataset(tmp_path)
    app['setup_review']()
    app['refresh_group_listboxes']()
    app['go_to_step']('organize')
    app['view_type_var'].set('layers')
    app['on_view_type_change']()
    app['optical_group_list'].selection_clear(0, 'end')
    app['on_group_selection']()
    app['review_display_var'].set('Máscara selecionada')
    app['change_preview']()
    assert app['session']['workspace_selected_groups'] == []
    assert not np.any(app['session']['current_rgb'])
    app['optical_group_list'].selection_set(0, 1)
    app['on_group_selection']()
    app['go_to_step']('refine')
    app['go_to_step']('results')
    assert app['selected_workspace_groups']() == [app['session']['prepared']['review'].selected_cluster]


@pytest.mark.parametrize('failure', ['history', 'workspace', 'reserved_array', 'corrupt',
                                     'widgets', 'preview', 'legend', 'save_previous'])
def test_failed_project_open_preserves_session_and_visual_state(app, tmp_path, monkeypatch, failure):
    import copy
    from test_riima_review import prepared_dataset, rewrite_project
    previous = prepared_dataset(tmp_path, count=2)
    session = app['session']
    session['prepared'] = previous
    session['project_path'] = str(tmp_path / 'previous.riima')
    session['images'] = [dict(path=item['path'], name=item['name']) for item in previous['review'].items]
    app['rebuild_image_rows']()
    app['setup_review']()
    app['refresh_group_listboxes']()
    app['go_to_step']('organize')
    app['optical_group_list'].selection_set(0, 1)
    app['on_group_selection']()
    app['review_opacity_var'].set(32)
    app['view_type_var'].set('compare')
    app['on_view_type_change']()
    session['split_fraction'] = .37
    app['change_preview']()
    session['_active_vars'][0].set(False)
    previous['review'].annotate(0, 0, [212], 1)
    review.save_project(previous, session['project_path'])
    incoming = tmp_path / 'incoming'
    incoming.mkdir()
    candidate = prepared_dataset(incoming, count=2)
    candidate['review'].selected_image = 1
    candidate_path = tmp_path / 'incoming.riima'
    review.save_project(candidate, candidate_path)
    if failure == 'history':
        rewrite_project(candidate_path, lambda arrays, meta: meta.update(output_history=[None]))
    elif failure == 'workspace':
        rewrite_project(candidate_path, lambda arrays, meta: meta.update(workspace={'selected_groups': None}))
    elif failure == 'reserved_array':
        rewrite_project(candidate_path, lambda arrays, meta: meta['items'][0]['optional_arrays'].append('labels'))
    elif failure == 'corrupt':
        candidate_path.write_bytes(b'not a project')
    elif failure == 'widgets':
        current = app['review_image_combo'].current
        armed = [True]
        def fail_current(index=None):
            if index == 1 and armed[0]:
                armed[0] = False
                raise RuntimeError('widget reconstruction failed')
            return current() if index is None else current(index)
        monkeypatch.setattr(app['review_image_combo'], 'current', fail_current)
    elif failure in ('preview', 'legend'):
        name = 'render_workspace' if failure == 'preview' else 'workspace_legend'
        original = getattr(viewer, name)
        def fail_candidate(prepared, *args, **kwargs):
            if prepared['review'] is not previous['review']:
                raise RuntimeError('candidate presentation failed')
            return original(prepared, *args, **kwargs)
        monkeypatch.setattr(viewer, name, fail_candidate)
    elif failure == 'save_previous':
        def fail_save(*args, **kwargs):
            raise OSError('disk full')
        monkeypatch.setattr(review, 'save_project', fail_save)
    monkeypatch.setattr(app['filedialog'], 'askopenfilename', lambda **kw: str(candidate_path))
    messages = []
    monkeypatch.setattr(app['messagebox'], 'showerror', lambda *args: messages.append(args))
    controls = ('out_var', 'automatic_output_var', 'reference_var', 'clusters_var', 'mask_mode_var',
                'workflow_var', 'review_tool_var', 'review_display_var', 'review_opacity_var',
                'review_image_var', 'review_cluster_var', 'view_type_var', 'split_left_var',
                'split_right_var', 'isolate_var', 'contours_var', 'show_applied_var',
                'status_var', 'summary_var', 'preview_title_var', 'review_status_var')
    before_controls = {key: app[key].get() for key in controls}
    before_workspace = copy.deepcopy(previous.get('workspace'))
    before_path = session['project_path']
    before_groups = session['workspace_selected_groups'][:]
    before_rgb = session['current_rgb'].copy()
    before_photo = session['photo']
    before_viewport = session['viewport']
    before_cache = session['compare_cache']
    before_active = [v.get() for v in session['_active_vars']]
    before_selection = app['optical_group_list'].curselection()
    before_frames = {key: frame.winfo_manager() for key, frame in app['step_frames'].items()}
    def legend_text(widget):
        return [(child.cget('text') if 'text' in child.keys() else '', legend_text(child))
                for child in widget.winfo_children()]
    before_legend = legend_text(app['legend_rows'])
    canvas = app['preview_canvas']
    def canvas_contents():
        return [(canvas.type(i), canvas.coords(i), canvas.itemconfigure(i)) for i in canvas.find_all()]
    before_canvas = canvas_contents()
    before_arrays = [{key: item[key].copy() for key in ('initial', 'labels', 'locked')}
                     for item in previous['review'].items]
    timers = {key: app['root'].after(60000, lambda: None) for key in ('review_after', 'autosave_after')}
    session.update(timers)
    try:
        app['open_project']()
        assert messages
        assert session['prepared'] is previous
        assert session['project_path'] == before_path
        assert previous.get('workspace') == before_workspace
        assert previous['review'].dirty == {0}
        assert session['workspace_selected_groups'] == before_groups
        assert session['viewport'] is before_viewport
        assert session['compare_cache'] is before_cache
        assert session['photo'] is before_photo
        np.testing.assert_array_equal(session['current_rgb'], before_rgb)
        assert {key: app[key].get() for key in controls} == before_controls
        assert [v.get() for v in session['_active_vars']] == before_active
        assert app['optical_group_list'].curselection() == before_selection
        assert {key: frame.winfo_manager() for key, frame in app['step_frames'].items()} == before_frames
        assert legend_text(app['legend_rows']) == before_legend
        assert canvas_contents() == before_canvas
        for item, arrays in zip(previous['review'].items, before_arrays):
            for key, values in arrays.items():
                np.testing.assert_array_equal(item[key], values)
        for key, timer in timers.items():
            assert session[key] == timer
            assert app['root'].tk.call('after', 'info', timer)
    finally:
        for key, timer in timers.items():
            app['root'].after_cancel(timer)
            session[key] = None


@pytest.mark.parametrize('failure', ['preview', 'legend'])
def test_failed_first_project_open_preserves_unanalyzed_images(app, tmp_path, monkeypatch, failure):
    from test_riima_review import prepared_dataset
    candidate = prepared_dataset(tmp_path)
    path = tmp_path / 'candidate.riima'
    review.save_project(candidate, path)
    session = app['session']
    session.update(prepared=None, project_path=None, workspace_selected_groups=[], compare_mode=False,
                   images=[dict(path=candidate['review'].items[0]['path'], name='Selected input')])
    app['view_type_var'].set('layers')
    app['rebuild_image_rows']()
    app['refresh_group_listboxes']()
    app['go_to_step']('images')
    session['_active_vars'][0].set(False)
    name = 'render_workspace' if failure == 'preview' else 'workspace_legend'
    active_seen = []
    def fail_candidate(prepared, *args, **kwargs):
        active_seen.append(session['prepared'])
        raise RuntimeError('first project reconstruction failed')
    monkeypatch.setattr(viewer, name, fail_candidate)
    monkeypatch.setattr(app['filedialog'], 'askopenfilename', lambda **kw: str(path))
    messages = []
    monkeypatch.setattr(app['messagebox'], 'showerror', lambda *args: messages.append(args))
    before_images = session['images'][:]
    canvas = app['preview_canvas']
    before_canvas = [(canvas.type(i), canvas.coords(i), canvas.itemconfigure(i)) for i in canvas.find_all()]
    before_status = app['status_var'].get()
    app['open_project']()
    assert messages and active_seen == [None]
    assert session['prepared'] is None and session['project_path'] is None
    assert session['images'] == before_images
    assert [v.get() for v in session['_active_vars']] == [False]
    assert session['current_rgb'] is None
    assert app['workflow_var'].get() == 'images'
    assert app['status_var'].get() == before_status
    assert [(canvas.type(i), canvas.coords(i), canvas.itemconfigure(i)) for i in canvas.find_all()] == before_canvas


@pytest.mark.parametrize('version', [1, 2, 3, 4])
def test_historical_project_open_restores_valid_workspace_without_changing_science(app, tmp_path, monkeypatch, version):
    from test_riima_review import prepared_dataset, rewrite_project
    prepared = prepared_dataset(tmp_path)
    prepared['review'].selected_image = 1
    prepared['interpretation'].rename_cluster(1, 'Preserved name')
    prepared['workspace'] = dict(selected_groups=[1], opacity=27., split_fraction=.25,
                                 step='organize', display='Máscara selecionada', view_type='compare',
                                 compare_left='Imagem original', compare_right='Máscara selecionada')
    path = tmp_path / 'historical.riima'
    review.save_project(prepared, path)
    def legacy(arrays, meta):
        meta['version'] = version
        for i in range(1, len(meta['items'])):
            arrays[f'{i}_sample_X'] = arrays['0_sample_X'].copy()
    rewrite_project(path, legacy)
    app['session'].update(prepared=None, project_path=None)
    monkeypatch.setattr(app['filedialog'], 'askopenfilename', lambda **kw: str(path))
    monkeypatch.setattr(app['messagebox'], 'showerror', lambda *args: pytest.fail(str(args)))
    app['open_project']()
    loaded = app['session']['prepared']
    assert app['session']['project_path'] == str(path)
    assert app['session']['workspace_selected_groups'] == [1]
    assert app['session']['split_fraction'] == .25
    assert app['workflow_var'].get() == 'organize'
    assert app['view_type_var'].get() == 'compare'
    assert app['review_opacity_var'].get() == 27.
    assert loaded['interpretation'].to_dict() == prepared['interpretation'].to_dict()
    for key in ('labels', 'centers', 'center', 'scale'):
        np.testing.assert_array_equal(loaded['optical'][key], prepared['optical'][key])
    for actual, expected in zip(loaded['review'].items, prepared['review'].items):
        for key in ('initial', 'labels', 'locked'):
            np.testing.assert_array_equal(actual[key], expected[key])


def test_open_active_project_keeps_unsaved_review_changes(app, tmp_path, monkeypatch):
    from test_riima_review import prepared_dataset
    prepared = prepared_dataset(tmp_path, count=1)
    path = tmp_path / 'active.riima'
    review.save_project(prepared, path)
    app['session'].update(prepared=prepared, project_path=str(path))
    app['setup_review']()
    app['refresh_group_listboxes']()
    app['go_to_step']('organize')
    prepared['review'].annotate(0, 0, [212], 1)
    prepared['interpretation'].rename_cluster(0, 'Unsaved name')
    monkeypatch.setattr(app['filedialog'], 'askopenfilename', lambda **kw: str(path))
    monkeypatch.setattr(app['messagebox'], 'showerror', lambda *args: pytest.fail(str(args)))
    app['open_project']()
    loaded = app['session']['prepared']
    assert loaded['review'].dirty == {0}
    assert loaded['review'].items[0]['annotations'][0].flat[212] == 1
    assert loaded['interpretation'].to_dict() == prepared['interpretation'].to_dict()
    np.testing.assert_array_equal(loaded['optical']['labels'], prepared['optical']['labels'])
    np.testing.assert_array_equal(loaded['optical']['centers'], prepared['optical']['centers'])


@pytest.fixture(scope="module")
def app():
    """One real widget tree shared by every test in this module.

    Building a fresh ``tk.Tk()`` per test is flaky on this platform: destroying one root
    and immediately creating another in the same process intermittently raises
    ``TclError: invalid command name "tcl_findLibrary"`` (a known CPython/Tcl teardown
    issue), even though each individual root works fine in isolation. Sharing one root and
    letting each test set only the ``session`` keys it needs avoids that entirely.
    """
    captured = {}

    def hook(ns):
        captured.update(ns)

    riima_gui.gui(_test_hook=hook)
    captured["root"].update()
    yield captured
    captured["root"].destroy()


@pytest.fixture(scope="module")
def analyzed_app(app, tmp_path_factory):
    """Run a real 3-image analysis through the actual GUI closures (not core directly),
    so downstream layer/legend state matches what a real interactive session produces."""
    tmp_path = tmp_path_factory.mktemp("gui_legend")
    ref = synthetic_registration_image()
    m1 = warp_test_image(ref, dx=4.0, dy=-3.0, angle=0.10)
    m2 = warp_test_image(ref, dx=-7.0, dy=2.0, angle=-0.15)
    paths = write_rgb_images(tmp_path, [ref, m1, m2])

    session = app["session"]
    session["images"] = [{"path": p, "name": Path(p).name} for p in paths]
    app["rebuild_image_rows"]()
    app["reference_var"].set(0)
    app["out_var"].set(str(tmp_path / "out"))
    app["clusters_var"].set(4)
    app["analyze"]()
    assert session["prepared"] is not None
    assert len(session["_active_vars"]) == 3
    return app


def _make_press_event(x):
    class _Evt:
        pass
    e = _Evt()
    e.x = x
    e.y = 0
    return e


def test_comparator_drag_reaches_true_0_percent_at_the_left_edge(app):
    """Dragging beyond the left image edge reaches split_fraction == 0.0."""
    session = app["session"]
    shape = (100, 200)  # (height, width)
    session["current_image_shape"] = shape
    session["viewport"] = viewer.ViewportState.fit(shape, (200, 100))
    session["compare_mode"] = True
    session["drag_mode"] = "divider"
    session["compare_cache"] = None  # drawing is skipped; only split_fraction math is under test

    app["on_canvas_drag"](_make_press_event(-500))  # far left of the canvas, well past the image
    assert session["split_fraction"] == 0.0


def test_comparator_drag_reaches_true_100_percent_at_the_right_edge(app):
    session = app["session"]
    shape = (100, 200)
    session["current_image_shape"] = shape
    session["viewport"] = viewer.ViewportState.fit(shape, (200, 100))
    session["compare_mode"] = True
    session["drag_mode"] = "divider"
    session["compare_cache"] = None

    app["on_canvas_drag"](_make_press_event(5000))  # far right, well past the image
    assert session["split_fraction"] == 1.0


def test_comparator_drag_is_not_clamped_to_the_old_005_095_range(app):
    """The comparison divider reaches both image edges."""
    session = app["session"]
    canvas = app["preview_canvas"]
    viewport_size = (max(1, canvas.winfo_width()), max(1, canvas.winfo_height()))
    shape = (100, 200)
    session["current_image_shape"] = shape
    viewport = viewer.ViewportState.fit(shape, viewport_size)
    session["viewport"] = viewport
    session["compare_mode"] = True
    session["drag_mode"] = "divider"
    session["compare_cache"] = None

    canvas_x_at_left_edge, _ = viewer.image_to_canvas((0.0, 0.0), viewport, shape, viewport_size)
    app["on_canvas_drag"](_make_press_event(int(canvas_x_at_left_edge)))
    assert session["split_fraction"] < 0.01

    canvas_x_at_right_edge, _ = viewer.image_to_canvas((float(shape[1]), 0.0), viewport, shape, viewport_size)
    app["on_canvas_drag"](_make_press_event(int(canvas_x_at_right_edge)))
    assert session["split_fraction"] > 0.99


def test_divider_handle_draws_only_a_line_and_a_circle_no_glyph(app):
    """The divider uses a line and circular handle without a text overlay."""
    session = app["session"]
    canvas = app["preview_canvas"]
    viewport_size = (max(1, canvas.winfo_width()), max(1, canvas.winfo_height()))
    shape = (100, 200)
    session["viewport"] = viewer.ViewportState.fit(shape, viewport_size)
    canvas.delete("all")  # drop the startup "Selecione imagens..." placeholder text first
    app["draw_divider_handle"](0.5, shape, viewport_size)
    items = canvas.find_all()
    types = [canvas.type(i) for i in items]
    assert "text" not in types
    assert "line" in types
    assert "oval" in types
    for i in items:
        if canvas.type(i) == "text":
            pytest.fail("no text item should be drawn on the divider handle")


def _collect_label_texts(widget):
    """Collect every Label's displayed text, including ``textvariable``-bound labels
    (whose ``cget("text")`` is always empty; the live value lives in the Tk variable)."""
    texts = []
    for child in widget.winfo_children():
        try:
            text = child.cget("text")
        except Exception:
            text = None
        if text:
            texts.append(str(text))
        try:
            varname = child.cget("textvariable")
        except Exception:
            varname = None
        if varname:
            try:
                texts.append(str(child.tk.globalgetvar(varname)))
            except Exception:
                pass
        texts.extend(_collect_label_texts(child))
    return texts


def test_workspace_legend_names_selected_image_and_current_groups(analyzed_app):
    app = analyzed_app
    app['go_to_step']('organize')
    app['view_type_var'].set('layers')
    app['on_view_type_change']()
    app['review_display_var'].set('Classificação')
    app['isolate_var'].set(False)
    app['change_preview']()
    texts = ' | '.join(_collect_label_texts(app['legend_rows']))
    assert app['review_image_var'].get() in texts
    for c in app['session']['prepared']['review'].class_ids:
        assert f'Grupo {c}' in texts
    assert 'Clusters brutos' not in texts
    assert app['layers_panel'].winfo_manager() == ''

def test_legend_comparator_names_original_and_current_classification(analyzed_app):
    app = analyzed_app
    app['view_type_var'].set('compare')
    app['on_view_type_change']()
    assert set(app['session']['_compare_sources']) == {'Imagem original', 'Classificação atual', 'Máscara selecionada'}
    app['split_left_var'].set('Imagem original')
    app['split_right_var'].set('Classificação atual')
    app['on_compare_source_change']()
    texts = ' | '.join(_collect_label_texts(app['legend_rows']))
    assert 'Esquerda: Imagem original' in texts
    assert app['review_image_var'].get() in texts
    assert 'Direita: Classificação atual' in texts
    for c in app['session']['prepared']['review'].class_ids:
        assert f'Grupo {c}' in texts

def test_comparator_selected_mask_follows_current_image_and_selection(analyzed_app):
    app = analyzed_app
    app['go_to_step']('organize')
    app['optical_group_list'].selection_clear(0, 'end')
    app['optical_group_list'].selection_set(1)
    app['on_group_selection']()
    app['view_type_var'].set('compare')
    app['on_view_type_change']()
    app['split_left_var'].set('Imagem original')
    app['split_right_var'].set('Máscara selecionada')
    app['on_compare_source_change']()
    texts = ' | '.join(_collect_label_texts(app['legend_rows']))
    assert 'Direita: Máscara selecionada' in texts
    assert 'Grupo 1' in texts
    assert 'Grupo 0' not in texts
    app['review_image_combo'].current(1)
    app['select_review_image']()
    prepared = app['session']['prepared']
    rgb = prepared['optical']['images'][1]
    np.testing.assert_array_equal(app['session']['compare_cache']['left_rgb'], rgb)
    mask = viewer.workspace_mask(prepared, [1])
    np.testing.assert_array_equal(app['session']['compare_cache']['right_rgb'][..., 0], mask.astype(np.uint8) * 255)
    app['view_type_var'].set('layers')
    app['on_view_type_change']()

def _collect_canvases(widget):
    out = []
    for child in widget.winfo_children():
        if isinstance(child, tk.Canvas):
            out.append(child)
        out.extend(_collect_canvases(child))
    return out


def _collect_treeviews(widget):
    from tkinter import ttk
    out = []
    for child in widget.winfo_children():
        if isinstance(child, ttk.Treeview):
            out.append(child)
        out.extend(_collect_treeviews(child))
    return out


def test_degenerate_diagnostics_are_unavailable_and_never_suggested(analyzed_app, monkeypatch):
    app = analyzed_app
    root = app['root']
    result = {
        'k_values': [2, 3], 'inertia': [0.0, 0.0],
        'silhouette': [0.8, 0.9], 'calinski_harabasz': [1.0, 1.0],
        'davies_bouldin': [0.0, 0.0], 'diagnostic_valid': [False, False],
        'diagnostic_warnings': [['Coincident centers'], ['Empty clusters']],
    }
    monkeypatch.setattr(core, 'evaluate_cluster_counts', lambda *a, **kw: result)
    before = set(root.winfo_children())
    app['open_cluster_count_diagnostics']()
    time.sleep(0.1)
    root.update()
    win, = [w for w in root.winfo_children() if isinstance(w, tk.Toplevel) and w not in before]
    try:
        tree, = _collect_treeviews(win)
        rows = [tree.item(item, 'values') for item in tree.get_children('')]
        assert all(row[2:] == ('n/d', 'n/d', 'n/d') for row in rows)
        assert all(not tree.item(item, 'tags') for item in tree.get_children(''))
        assert any('degenerad' in text.lower() for text in _collect_label_texts(win))
        assert any('sem sugestão' in text.lower() for text in _collect_label_texts(win))
        for canvas in _collect_canvases(win)[1:]:
            assert not any(canvas.type(item) == 'oval' for item in canvas.find_all())
        assert result['davies_bouldin'] == [0.0, 0.0]
        assert result['silhouette'] == [0.8, 0.9]
    finally:
        win.destroy()


def test_diagnostics_dialog_shows_suggested_k_and_charts_without_recomputing_on_resize(analyzed_app, monkeypatch):
    """The diagnostics dialog computes and displays an explicit
    suggested K (never claimed as mineralogically correct), draw the 4 metric line
    charts, and never recompute the scientific metrics just because a chart is resized."""
    app = analyzed_app
    root = app["root"]

    calls = {"n": 0}
    real_evaluate = core.evaluate_cluster_counts

    def counting_evaluate(*a, **kw):
        calls["n"] += 1
        return real_evaluate(*a, **kw)

    monkeypatch.setattr(core, "evaluate_cluster_counts", counting_evaluate)

    before_toplevels = set(root.winfo_children())
    app["open_cluster_count_diagnostics"]()
    time.sleep(0.1)
    root.update()  # let the dialog's win.after(50, run_evaluation) fire

    new_toplevels = [w for w in root.winfo_children() if isinstance(w, tk.Toplevel) and w not in before_toplevels]
    assert len(new_toplevels) == 1, "diagnostics dialog window was not created"
    win = new_toplevels[0]

    assert calls["n"] == 1

    texts = _collect_label_texts(win)
    assert any("sugerido" in t.lower() for t in texts)
    # The introduction distinguishes the suggested K from petrographic interpretation.
    assert sum(1 for t in texts if "sugerido" in t.lower()) >= 2  # intro text + "K = X (sugerido)"
    assert any("interpretação petrográfica" in t.lower() for t in texts)

    canvases = _collect_canvases(win)
    assert len(canvases) == 4
    for c in canvases:
        assert len(c.find_all()) > 0

    trees = _collect_treeviews(win)
    assert len(trees) == 1
    suggested_rows = [i for i in trees[0].get_children("") if "suggested" in trees[0].item(i, "tags")]
    assert len(suggested_rows) == 1
    assert trees[0].tag_configure('suggested')['background'] == '#098842'

    calls["n"] = 0
    for c in canvases:
        c.event_generate("<Configure>", width=222, height=111)
    root.update()
    assert calls["n"] == 0, "resizing a diagnostics chart must never recompute the scientific metrics"
    for c in canvases:
        assert len(c.find_all()) > 0  # still redrawn from the cached result

    prepared_before = app["session"]["prepared"]
    raw_before = prepared_before["optical"]["labels"].copy()
    from tkinter import ttk
    def find_suggestion(widget):
        for child in widget.winfo_children():
            if isinstance(child, ttk.Button) and child.cget("text") == "Usar K sugerido":
                return child
            found = find_suggestion(child)
            if found is not None:
                return found
        return None
    find_suggestion(win).invoke()
    assert app["session"]["prepared"] is prepared_before
    assert np.array_equal(prepared_before["optical"]["labels"], raw_before)
    assert calls["n"] == 0


def _text_of(widget):
    try:
        return widget.cget("text")
    except Exception:
        return None


def test_sidebar_shows_only_active_step_and_preserves_viewport(analyzed_app):
    app = analyzed_app
    app['zoom_button'](1.25)
    viewport = dataclasses.asdict(app['session']['viewport'])
    for step in ('classify', 'organize', 'refine', 'results'):
        app['go_to_step'](step)
        assert [key for key, frame in app['step_frames'].items() if frame.winfo_manager()] == [step]
        assert dataclasses.asdict(app['session']['viewport']) == viewport
    assert app['diagnostics_slot'].master is app['step_frames']['classify']
    assert app['optical_group_frame'].master is app['step_frames']['organize']
    assert app['review_frame'].master is app['step_frames']['refine']

def test_diagnostics_block_is_compact_not_a_tall_bordered_section(app):
    """The diagnostics entry fits within the classification sidebar."""
    from tkinter import ttk
    diagnostics_frame = app["diagnostics_frame"]
    assert not isinstance(diagnostics_frame, ttk.Labelframe)
    children = diagnostics_frame.winfo_children()
    buttons = [w for w in children if isinstance(w, ttk.Button)]
    assert len(buttons) == 1
    assert buttons[0].cget("text") == "Diagnóstico de K..."


def test_cluster_count_spinbox_and_diagnostic_share_the_same_k_range(app):
    """The main K spinner and the on-demand diagnostic use the
    exact same K range, so the diagnostic can never suggest a K the spinner can't hold."""
    from tkinter import ttk

    def find_spinbox(widget):
        for child in widget.winfo_children():
            if isinstance(child, ttk.Spinbox):
                return child
            found = find_spinbox(child)
            if found is not None:
                return found
        return None

    spin = find_spinbox(app["sidebar"])
    assert spin is not None
    assert int(float(spin.cget("from"))) == 2
    assert int(float(spin.cget("to"))) == 12


def test_header_contains_product_identity_and_project_actions(app):
    texts = _collect_label_texts(app['header'])
    assert texts[:2] == ['RIIMA', 'Really Intuitive Image Mapping by Affinity']
    assert texts[2:] == ['Abrir projeto...', 'Salvar projeto']

def test_manual_cluster_family_actions_preserve_raw_and_refresh_comparator(analyzed_app, monkeypatch):
    app = analyzed_app
    prepared = app["session"]["prepared"]
    raw = prepared["optical"]["labels"].copy()
    interpretation = prepared["interpretation"]
    monkeypatch.setattr(app["simpledialog"], "askstring", lambda *a, **kw: "Feldspato")
    app["optical_group_list"].selection_clear(0, "end")
    app["optical_group_list"].selection_set(0)
    app["optical_group_list"].selection_set(2)
    app["create_family_selected"]()
    assert interpretation.families[0]["clusters"] == [0, 2]
    app['optical_group_list'].selection_clear(0, 'end')
    app["optical_group_list"].selection_set(0)
    app["optical_group_list"].selection_set(2)
    app["merge_optical_selected"]()
    assert {0, 2} in interpretation.groups
    assert np.array_equal(prepared["optical"]["labels"], raw)
    assert set(app["session"]["_compare_sources"]) == {"Imagem original", "Classificação atual", "Máscara selecionada"}
    app["apply_interpretation"]("reset")
    assert interpretation.groups == [{i} for i in range(prepared["optical"]["clusters"])]
    assert interpretation.families == []


def test_gui_and_cli_do_not_depend_on_removed_pipeline(monkeypatch):
    import ast
    source = Path(riima_gui.__file__).read_text(encoding="utf-8")
    calls = {node.attr for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Attribute)}
    assert not any("grain" in name or "watershed" in name for name in calls)
    monkeypatch.setattr("sys.argv", ["riima_gui.py", "--clusters", "2"])
    args = riima_gui.parse_args()
    assert args.clusters == 2
    assert not hasattr(args, "merge_strength")
    assert not hasattr(args, "grain_families")


def test_saved_association_metadata_preserved_without_an_extra_product(analyzed_app, monkeypatch):
    app = analyzed_app
    interpretation = app['session']['prepared']['interpretation']
    app['apply_interpretation']('reset')
    app['apply_interpretation']('create_family', [0], 'A')
    app['go_to_step']('organize')
    app['family_group_list'].selection_set(0)
    app['optical_group_list'].selection_clear(0, 'end')
    app['optical_group_list'].selection_set(1)
    app['change_family_membership']('add_to_family')
    assert interpretation.families[0]['clusters'] == [0, 1]
    app['family_group_list'].selection_set(0)
    app['select_legacy_mask']()
    assert app['session']['workspace_selected_groups'] == [0, 1]
    app['family_group_list'].selection_set(0)
    monkeypatch.setattr(app['simpledialog'], 'askstring', lambda *a, **kw: 'Plagioclásio')
    app['rename_selected_group'](app['family_group_list'], 'family')
    assert 'Plagioclásio' in app['family_group_list'].get(0)
    assert 'Famílias' not in app['session']['_compare_sources']
    app['family_group_list'].selection_set(0)
    app['delete_family_selected']()
    assert interpretation.families == []
    app['undo_optical_groups']()
    assert interpretation.families[0]['name'] == 'Plagioclásio'
    app['apply_interpretation']('reset')

def test_gui_reports_cross_family_merge_without_changing_state(analyzed_app, monkeypatch):
    app = analyzed_app
    interpretation = app["session"]["prepared"]["interpretation"]
    app["apply_interpretation"]("reset")
    app["apply_interpretation"]("create_family", [0], "A")
    app["apply_interpretation"]("create_family", [1], "B")
    before = interpretation.to_dict()
    messages = []
    monkeypatch.setattr(app["messagebox"], "showinfo", lambda title, text: messages.append(text))
    app["optical_group_list"].selection_set(0, 1)
    app["merge_optical_selected"]()
    assert interpretation.to_dict() == before
    assert "famílias diferentes" in messages[0]
    app["apply_interpretation"]("reset")


@pytest.mark.parametrize('failure', ['second_image', 'review', 'publication'])
def test_export_autosave_rolls_back_project_and_all_images(app, tmp_path, monkeypatch, failure):
    from test_riima_review import prepared_dataset
    from test_export_integrity import file_bytes, scientific_state
    from riima import exports
    prepared = prepared_dataset(tmp_path)
    session = app['session']
    path = str(tmp_path / 'session.riima')
    session.update(prepared=prepared, project_path=path)
    app['setup_review']()
    assert app['autosave_current']()
    before = file_bytes(tmp_path)
    prepared['interpretation'].rename_cluster(0, 'Changed')
    state = scientific_state(prepared)
    workspace = prepared['workspace']
    messages = []
    monkeypatch.setattr(app['messagebox'], 'showerror', lambda *args: messages.append(args))
    with monkeypatch.context() as patch:
        if failure == 'second_image':
            original = core.save_session_outputs
            def fail(data):
                if str(data['output']).endswith('image_001'):
                    raise OSError('second image failed')
                return original(data)
            patch.setattr(core, 'save_session_outputs', fail)
        elif failure == 'review':
            patch.setattr(review, 'save_review_outputs', lambda *args: (_ for _ in ()).throw(OSError('review failed')))
        else:
            original = exports.os.replace
            def fail(source, target):
                if '.riima-export' in str(source) and Path(target).name == 'statistics.csv':
                    raise OSError('publication failed')
                return original(source, target)
            patch.setattr(exports.os, 'replace', fail)
        assert app['autosave_current']() is False
    assert messages and session['project_path'] == path
    assert prepared['workspace'] is workspace
    assert scientific_state(prepared) == state
    assert file_bytes(tmp_path) == before
    assert app['autosave_current']()


def test_metadata_write_failure_keeps_previous_json_and_scientific_state(app, tmp_path, monkeypatch):
    from test_riima_review import prepared_dataset
    from test_export_integrity import fail_writes, scientific_state
    prepared = prepared_dataset(tmp_path)
    app['session'].update(prepared=prepared, project_path=None)
    app['save_interpretation_metadata']()
    target = Path(prepared['output']) / 'interpretation.json'
    before = target.read_bytes()
    prepared['interpretation'].rename_cluster(0, 'Updated metadata')
    state = scientific_state(prepared)
    with monkeypatch.context() as patch:
        fail_writes(patch, 'json')
        patch.setattr(core, 'save_session_outputs', lambda *args: pytest.fail('full export'))
        app['save_interpretation_metadata']()
    assert target.read_bytes() == before
    assert scientific_state(prepared) == state
    app['save_interpretation_metadata']()
    assert target.read_bytes() != before


@pytest.mark.parametrize('failure', ['review', 'publication'])
def test_analysis_autosave_failure_keeps_previous_scientific_files(app, tmp_path, monkeypatch, failure):
    from test_riima_review import prepared_dataset
    from test_export_integrity import file_bytes
    prepared = prepared_dataset(tmp_path, count=1)
    path = prepared['review'].items[0]['path']
    session = app['session']
    session.update(prepared=prepared, project_path=str(tmp_path / 'checkpoint.riima'),
                   images=[dict(path=path, name=Path(path).name)])
    app['setup_review']()
    assert app['autosave_current']()
    before = file_bytes(Path(prepared['output']))
    app['out_var'].set(prepared['output'])
    app['clusters_var'].set(3)
    app['reference_var'].set(0)
    app['rebuild_image_rows']()
    messages = []
    monkeypatch.setattr(app['messagebox'], 'showerror', lambda *args: messages.append(args))
    if failure == 'review':
        def fail(*args):
            raise OSError('review export failed')
        monkeypatch.setattr(review, 'save_review_outputs', fail)
    else:
        from riima import exports
        original = exports.os.replace
        def fail(source, target):
            if Path(target).name == 'statistics.csv' and '.riima-export' in str(source):
                raise OSError('publication failed')
            return original(source, target)
        monkeypatch.setattr(exports.os, 'replace', fail)
    app['analyze']()
    assert messages
    assert session['prepared'] is not prepared
    assert session['prepared']['config'].clusters == 3
    assert file_bytes(Path(prepared['output'])) == before
    assert 'pendente' in app['status_var'].get().lower()


def test_choose_output_changes_the_loaded_export_destination(app, tmp_path, monkeypatch):
    from test_riima_review import prepared_dataset
    prepared = prepared_dataset(tmp_path)
    path = tmp_path / 'portable.riima'
    review.save_project(prepared, path)
    loaded = review.load_project(path)
    app['session'].update(prepared=loaded, project_path=str(path))
    selected = tmp_path / 'selected'
    selected.mkdir()
    monkeypatch.setattr(app['filedialog'], 'askdirectory', lambda **kw: str(selected))
    app['choose_out']()
    assert loaded['output'] == str(selected)
    assert loaded['output_pending'] is False
    assert app['out_var'].get() == str(selected)
    assert loaded['output_history'] == [prepared['output']]


@pytest.mark.parametrize('cancel', [False, True])
@pytest.mark.parametrize('denied_step', ['mkdir', 'probe'])
def test_resumed_readonly_destination_can_be_changed_or_cancelled_before_export(app, tmp_path, monkeypatch, cancel, denied_step):
    from test_riima_review import prepared_dataset
    prepared = prepared_dataset(tmp_path)
    path = tmp_path / 'readonly.riima'
    review.save_project(prepared, path)
    before_project = path.read_bytes()
    loaded = review.load_project(path)
    app['session'].update(prepared=loaded, project_path=str(path))
    app['out_var'].set(loaded['output'])
    app['setup_review']()
    initial = loaded['output']
    selected = tmp_path / 'writable'
    selected.mkdir()
    mkdir = Path.mkdir
    def deny(target, *args, **kwargs):
        if str(target) == initial and denied_step == 'mkdir':
            raise PermissionError('readonly parent')
        return mkdir(target, *args, **kwargs)
    monkeypatch.setattr(Path, 'mkdir', deny)
    temporary = riima_gui.tempfile.TemporaryFile
    def deny_write(*args, **kwargs):
        if str(kwargs.get('dir')) == initial and denied_step == 'probe':
            raise PermissionError('readonly output directory')
        return temporary(*args, **kwargs)
    monkeypatch.setattr(riima_gui.tempfile, 'TemporaryFile', deny_write)
    monkeypatch.setattr(app['messagebox'], 'showinfo', lambda *args: None)
    monkeypatch.setattr(app['messagebox'], 'showerror', lambda *args: pytest.fail(str(args)))
    monkeypatch.setattr(app['filedialog'], 'askdirectory', lambda **kw: '' if cancel else str(selected))
    result = app['autosave_current']()
    if cancel:
        assert result is False
        assert loaded['output'] == initial
        assert loaded['output_pending'] is True
        assert app['out_var'].get() == initial
        assert not list(selected.iterdir())
        assert path.read_bytes() == before_project
    else:
        assert result is True
        assert loaded['output'] == str(selected)
        assert loaded['output_pending'] is False
        assert (selected / 'review' / 'statistics.csv').exists()
    assert app['session']['prepared'] is loaded
    assert not Path(initial).exists()


def test_resumed_destination_created_by_another_writer_is_not_overwritten(app, tmp_path):
    from test_riima_review import prepared_dataset
    path = tmp_path / 'collision.riima'
    review.save_project(prepared_dataset(tmp_path), path)
    loaded = review.load_project(path)
    first = Path(loaded['output'])
    first.mkdir()
    (first / 'keep').write_text('other writer')
    app['session'].update(prepared=loaded, project_path=str(path))
    app['setup_review']()
    assert app['autosave_current']()
    assert Path(loaded['output']) == tmp_path / 'collision_output_2'
    assert sorted(p.name for p in first.iterdir()) == ['keep']


@pytest.mark.parametrize('exporter', ['export_light_metrics', 'export_descriptive_report'])
@pytest.mark.parametrize('scenario', ['collision', 'permission', 'cancel'])
def test_results_exports_claim_portable_destination_before_writing(app, tmp_path, monkeypatch, exporter, scenario):
    from riima import metrics, report
    from test_riima_review import prepared_dataset
    from test_export_integrity import scientific_state, file_bytes
    original = prepared_dataset(tmp_path)
    old_output = Path(original['output'])
    old_output.mkdir()
    (old_output / 'keep').write_text('previous results')
    project = tmp_path / 'resume.riima'
    review.save_project(original, project)
    loaded = review.load_project(project)
    initial = Path(loaded['output'])
    selected = tmp_path / 'selected'
    selected.mkdir()
    if scenario == 'collision':
        initial.mkdir()
        (initial / 'keep').write_text('another writer')
    else:
        mkdir = Path.mkdir
        def deny(target, *args, **kwargs):
            if target == initial:
                raise PermissionError('readonly destination')
            return mkdir(target, *args, **kwargs)
        monkeypatch.setattr(Path, 'mkdir', deny)
    app['session'].update(prepared=loaded, project_path=str(project))
    app['out_var'].set(str(initial))
    app['setup_review']()
    state = scientific_state(loaded)
    viewport = app['session']['viewport']
    before_project, before_old = project.read_bytes(), file_bytes(old_output)
    output_report = tmp_path / 'report.html'
    monkeypatch.setattr(app['filedialog'], 'asksaveasfilename', lambda **kw: str(output_report))
    monkeypatch.setattr(app['filedialog'], 'askdirectory', lambda **kw: '' if scenario == 'cancel' else str(selected))
    monkeypatch.setattr(app['messagebox'], 'showinfo', lambda *args: None)
    monkeypatch.setattr(app['messagebox'], 'showerror', lambda *args: pytest.fail(str(args)))
    calls = []
    def write_export(data, path=None, **kwargs):
        calls.append(data)
        (Path(data['output']) / 'export-marker').write_text('exported')
        if path is not None:
            Path(path).write_text('report')
            return Path(path)
        return Path(data['output'])
    if exporter == 'export_light_metrics':
        monkeypatch.setattr(metrics, 'export_metrics', write_export)
    else:
        monkeypatch.setattr(report, 'export_report', write_export)
    app[exporter]()
    assert app['session']['prepared'] is loaded
    assert app['session']['viewport'] is viewport
    assert app['session']['project_path'] == str(project)
    assert scientific_state(loaded) == state
    assert loaded['output_history'] == [original['output']]
    assert project.read_bytes() == before_project
    assert file_bytes(old_output) == before_old
    if scenario == 'cancel':
        assert calls == []
        assert loaded['output_pending'] is True
        assert loaded['output'] == str(initial) == app['out_var'].get()
        assert not initial.exists() and not output_report.exists()
        assert not list(selected.iterdir())
    else:
        expected = tmp_path / 'resume_output_2' if scenario == 'collision' else selected
        assert calls == [loaded]
        assert loaded['output_pending'] is False
        assert Path(loaded['output']) == expected == Path(app['out_var'].get())
        assert (expected / 'export-marker').read_text() == 'exported'
        if scenario == 'collision':
            assert sorted(p.name for p in initial.iterdir()) == ['keep']
        else:
            assert not initial.exists()


# Shared visual identity contracts use the same real widget tree as this module.

def test_tk_font_fallback_is_selected_from_installed_families(app, monkeypatch):
    from riima import theme
    from tkinter import font as tkfont
    root, style = app['root'], app['style']
    original = tkfont.families
    try:
        monkeypatch.setattr(tkfont, 'families', lambda root=None: ('Arial', 'Noto Sans'))
        theme.apply(root, style)
        assert theme.FONT_FAMILY == 'Noto Sans'
        # The synthetic inventory exercises selection; Tk may resolve this family
        # to a physical fallback when it is not actually installed on the host.
        assert tkfont.nametofont('TkDefaultFont', root=root).cget('family') == 'Noto Sans'
        monkeypatch.setattr(tkfont, 'families', lambda root=None: ())
        theme.apply(root, style)
        assert theme.FONT_FAMILY == 'sans-serif'
    finally:
        monkeypatch.setattr(tkfont, 'families', original)
        theme.apply(root, style)



def test_native_listbox_combobox_and_table_follow_desktop_tokens(app):
    from riima import theme
    root, style = app['root'], app['style']
    listbox = app['optical_group_list']
    assert listbox.cget('background').lower() == theme.BG_PANEL.lower()
    assert listbox.cget('foreground').lower() == theme.TEXT_PRIMARY.lower()
    assert listbox.cget('selectbackground').lower() == theme.ACCENT.lower()
    assert listbox.cget('selectforeground').lower() == theme.TEXT_ON_ACCENT.lower()
    assert style.lookup('TCombobox', 'fieldbackground', ('readonly',)).lower() == theme.BG_PANEL.lower()
    assert style.lookup('TCombobox', 'foreground', ('readonly',)).lower() == theme.TEXT_PRIMARY.lower()
    assert style.lookup('Treeview', 'fieldbackground').lower() == theme.BG_PANEL.lower()
    assert style.lookup('Treeview', 'background', ('selected',)).lower() == theme.ACCENT.lower()
    assert style.lookup('Treeview.Heading', 'font')
    assert app['legend_canvas'].cget('background').lower() == theme.BG_APP.lower()
    # ttk Combobox popup listboxes are created lazily by Tcl and need option defaults.
    popup = tk.Listbox(root, name='visual_test_popup')
    try:
        assert popup.cget('selectbackground').lower() == theme.ACCENT.lower()
    finally:
        popup.destroy()



def test_diagnostic_canvas_text_and_marks_are_legible(app):
    from riima import theme
    from test_visual_desktop import _contrast
    canvas = tk.Canvas(app['root'], width=440, height=130, bg=theme.BG_CANVAS)
    canvas.pack()
    app['root'].update()
    try:
        for values in ([float('nan')], [.2, .6, .4]):
            app['_draw_metric_chart'](canvas, [2, 3, 4], values, 'Diagnóstico', 3)
            for item in canvas.find_all():
                fill = canvas.itemcget(item, 'fill')
                minimum = 4.5 if canvas.type(item) == 'text' else 3
                assert _contrast(fill, theme.BG_CANVAS) >= minimum, (canvas.type(item), fill)
    finally:
        canvas.destroy()



@pytest.mark.parametrize('geometry', ['1080x700', '1280x820'])
@pytest.mark.parametrize('label_padding', [None, (16, 3, 12, 3)])
def test_long_legend_label_wraps_inside_scrollable_viewport(app, geometry, label_padding):
    from tkinter import ttk
    root = app['root']
    style = app['style']
    before = root.geometry()
    before_padding = style.configure('TLabel', 'padding')
    long_label = 'Grupo de interpretação com nome muito extenso ' * 7
    try:
        if label_padding is not None:
            style.configure('TLabel', padding=label_padding)
        root.geometry(geometry)
        app['clear_legend']()
        app['_draw_legend_rows']('test', [dict(label=long_label, detail='Detalhes completos ' * 12, color=(120, 80, 220))])
        root.update()
        labels = [child for line in app['legend_rows'].winfo_children()
                  for frame in line.winfo_children() if isinstance(frame, ttk.Frame)
                  for child in frame.winfo_children() if isinstance(child, ttk.Label)]
        assert labels[0].cget('text') == long_label
        for label in labels:
            assert int(label.cget('wraplength')) > 0
            assert label.winfo_reqwidth() <= label.master.winfo_width() + 1
        assert labels[0].winfo_reqheight() > 30
        assert app['legend_canvas'].bbox('all')[3] >= labels[0].winfo_reqheight()
    finally:
        style.configure('TLabel', padding=before_padding or '')
        root.geometry(before)
        app['draw_legend']()

