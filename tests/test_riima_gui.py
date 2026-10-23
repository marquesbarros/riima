"""Integration tests for the native Tkinter widgets in ``riima.gui``.

``riima_gui.gui()`` accepts an optional ``_test_hook`` that is called with the
function's full local namespace right after the real widget tree is built, instead of
entering ``mainloop``. This drives the actual widgets (real Canvas, real event handler
closures, real ``session`` state dict) rather than mocking the GUI away, without blocking
on an event loop or requiring user interaction. Native desktop control tests reuse
this interpreter with isolated Toplevel hosts, avoiding repeated Tk initialization.
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


def _select_group_rows(app, first, last=None):
    rows = app['group_tree'].get_children()
    app['group_tree'].selection_add(rows[int(first):int(last if last is not None else first) + 1])


def _group_row_indices(app):
    rows = app['group_tree'].get_children()
    return tuple(rows.index(iid) for iid in app['group_tree'].selection())


def _group_texts(app):
    tree = app['group_tree']
    return tuple(tree.item(iid, 'text') + ' ' + str(tree.item(iid, 'values')) for iid in tree.get_children())


@pytest.fixture(autouse=True)
def confirm_replacement_by_default(app, monkeypatch):
    """Legacy workflows explicitly consent to replacement unless a test overrides it."""
    monkeypatch.setattr(app['messagebox'], 'askyesno', lambda *args, **kwargs: True)
    app['analysis_mode_var'].set('Registradas (mesmo campo)')
    for kind in ('showinfo', 'showwarning', 'showerror'):
        monkeypatch.setattr(app['messagebox'], kind, lambda *args, **kwargs: pytest.fail(f'Unexpected dialog: {args}'))


def test_two_environments_share_one_group_tree_and_independent_active_group(app, tmp_path):
    from tkinter import ttk
    from test_riima_review import prepared_dataset
    assert 'step_buttons' not in app and 'stepbar' not in app
    assert isinstance(app['group_tree'], ttk.Treeview)
    prepared = prepared_dataset(tmp_path)
    app['session']['prepared'] = prepared
    app['setup_review']()
    app['go_to_step']('interpretation')
    active = prepared['review'].selected_cluster
    app['select_groups']([1])
    assert prepared['review'].selected_cluster == active
    assert app['selected_workspace_groups']() == [1]
    assert app['group_tree'].selection() == ('1',)
    app['activate_selected_group']()
    assert prepared['review'].selected_cluster == 1
    assert app['review_frame'].winfo_manager() == 'pack'
    assert app['optical_group_frame'].winfo_manager() == 'pack'
    assert app['sidebar_container'].winfo_width() >= 252
    for button in app['review_tool_buttons'].values():
        assert button.cget('text') and str(button.cget('compound')) == 'image'
    app['go_to_step']('preparation')
    assert app['session']['prepared'] is prepared


def test_new_input_staging_and_cancelled_replacement_preserve_current_analysis(app, tmp_path, monkeypatch):
    from test_riima_review import prepared_dataset
    from test_export_integrity import scientific_state
    prepared = prepared_dataset(tmp_path)
    app['session'].update(prepared=prepared, project_path=str(tmp_path / 'previous.riima'))
    app['setup_review']()
    app['save_project_current']()
    before = scientific_state(prepared)
    archive = Path(app['session']['project_path']).read_bytes()
    monkeypatch.setattr(app['filedialog'], 'askopenfilenames', lambda **kw: [prepared['review'].items[0]['path']])
    app['choose_images']()
    assert app['session']['prepared'] is prepared
    assert scientific_state(prepared) == before
    monkeypatch.setattr(app['messagebox'], 'askyesno', lambda *a, **kw: False)
    app['analyze']()
    assert app['session']['prepared'] is prepared
    assert Path(app['session']['project_path']).read_bytes() == archive


@pytest.mark.parametrize('aspect', ['actions', 'active_name', 'scroll', 'sidebar_scroll', 'preparation_labels'])
def test_compact_interpretation_controls_fit_with_many_and_long_named_groups(app, tmp_path, aspect):
    from test_riima_review import prepared_dataset
    prepared = prepared_dataset(tmp_path)
    state = prepared['review']
    long_name = ('Grupo de aparência óptica com nome extenso ' * 4).strip()
    state.rename_group(state.selected_cluster, long_name)
    for index in range(15):
        state.create_group(f'Grupo adicional {index}')
    app['session']['prepared'] = prepared
    app['setup_review']()
    app['root'].geometry('1080x700')
    app['go_to_step']('interpretation')
    app['root'].update()
    if aspect == 'actions':
        frame = app['manager_actions']
        for button in frame.winfo_children():
            assert button.winfo_x() + button.winfo_reqwidth() <= frame.winfo_width()
            assert button.winfo_width() >= 28
        assert app['background_actions'] is app['selection_actions']
    elif aspect == 'active_name':
        label = app['active_group_label']
        assert label.winfo_height() <= 45
        tip = app['tooltips'][label]['text']
        assert long_name in (tip() if callable(tip) else tip)
        assert state.group_name(state.selected_cluster) == long_name
    elif aspect == 'scroll':
        tree = app['group_tree']
        assert tree.cget('yscrollcommand')
        tree.event_generate('<MouseWheel>', delta=-120)
        app['root'].update()
        assert tree.yview()[0] > 0
    elif aspect == 'sidebar_scroll':
        scrollbar = app['sidebar_scroll']
        canvas = app['sidebar_canvas']
        container = app['sidebar_container']
        assert scrollbar.winfo_ismapped()
        assert scrollbar.winfo_width() >= 10
        assert canvas.winfo_width() + scrollbar.winfo_width() <= container.winfo_width()
        assert scrollbar.winfo_x() + scrollbar.winfo_width() <= container.winfo_width()
        app['set_section_open']('import', True)
        app['set_section_open']('configuration', True)
        app['root'].update()
        canvas.yview_moveto(1)
        app['root'].update()
        approval = app['review_approve_btn']
        approval.focus_force()
        app['root'].update()
        assert canvas.winfo_rooty() <= approval.winfo_rooty()
        assert approval.winfo_rooty() + approval.winfo_height() <= canvas.winfo_rooty() + canvas.winfo_height()
        canvas.yview_moveto(0)
    else:
        from tkinter import ttk
        app['go_to_step']('preparation')
        app['root'].update()
        app['set_section_open']('configuration', True)
        app['root'].update()
        preparation = app['section_frames']['configuration']
        descriptions = [child for child in preparation.winfo_children()
                        if isinstance(child, ttk.Label) and int(child.cget('wraplength') or 0) > 0]
        assert len(descriptions) == 2
        assert all(label.winfo_reqwidth() <= preparation.winfo_width() for label in descriptions)


def test_native_group_catalog_crud_and_undo_restore_selection(app, tmp_path, monkeypatch):
    from test_riima_review import prepared_dataset
    prepared = prepared_dataset(tmp_path)
    app['session']['prepared'] = prepared
    app['setup_review']()
    app['go_to_step']('interpretation')
    app['select_groups']([0, 1])
    original = [item['labels'].copy() for item in prepared['review'].items]
    monkeypatch.setattr(app['simpledialog'], 'askstring', lambda *a, **kw: 'Novo grupo')
    app['create_group_current']()
    state = prepared['review']
    created = state.selected_cluster
    assert state.groups[created] == set()
    assert app['group_tree'].exists(str(created))
    assert 'vazio' in str(app['group_tree'].item(str(created), 'values'))
    assert app['selected_workspace_groups']() == [created]
    app['delete_group_current']()
    assert created not in state.groups
    app['undo_group_current']()
    assert created in state.groups and app['selected_workspace_groups']() == [created]
    app['undo_group_current']()
    assert created not in state.groups and app['selected_workspace_groups']() == [0, 1]
    for item, labels in zip(state.items, original):
        np.testing.assert_array_equal(item['labels'], labels)


def test_similarity_adds_selection_and_updates_names_without_recalculation(app, tmp_path):
    from tkinter import ttk
    from test_riima_review import prepared_dataset
    prepared = prepared_dataset(tmp_path)
    app['session']['prepared'] = prepared
    app['setup_review']()
    app['go_to_step']('interpretation')
    app['select_groups']([0])
    active = prepared['review'].selected_cluster
    window = app['open_group_similarity']()
    cached = app['session']['similarity_cache']
    tree = next(child for child in window.winfo_children()[0].winfo_children() if isinstance(child, ttk.Treeview))
    tree.selection_set('1')
    tree.event_generate('<<TreeviewSelect>>')
    assert app['selected_workspace_groups']() == [0, 1]
    assert prepared['review'].selected_cluster == active
    app['group_edit']('rename_group', 1, 'Nome em tempo real', presentation=True)
    assert window.winfo_exists()
    assert tree.item('1', 'text') == 'Nome em tempo real'
    assert app['session']['similarity_cache'] is cached
    window.destroy()


def test_official_logo_rotates_in_ui_thread_and_stops_after_operation(app):
    from tkinter import ttk
    import threading
    assert isinstance(app['operation_progress'], ttk.Label)
    observations = []
    release = threading.Event()
    def sample():
        observations.append((str(app['operation_progress'].cget('image')), app['session']['busy']))
        if len(observations) < 3:
            app['root'].after(70, sample)
        else:
            release.set()
    app['root'].after(25, sample)
    app['run_operation']('Animando logotipo', lambda: release.wait(5))
    assert len({image for image, busy in observations if busy}) >= 2
    assert app['session']['operation_after'] is None
    assert not app['operation_progress'].winfo_ismapped()


def test_replacement_project_failure_keeps_prior_analysis_and_archive(app, tmp_path, monkeypatch):
    from test_riima_review import prepared_dataset
    from test_export_integrity import scientific_state
    prepared = prepared_dataset(tmp_path, count=1)
    app['session'].update(prepared=prepared, project_path=None,
                          images=[dict(path=prepared['review'].items[0]['path'], name='field_0.png')])
    app['setup_review']()
    app['go_to_step']('preparation')
    app['out_var'].set(prepared['output'])
    app['rebuild_image_rows']()
    assert app['save_project_current']()
    archive = Path(app['session']['project_path'])
    previous_bytes, before = archive.read_bytes(), scientific_state(prepared)
    app['clusters_var'].set(3)
    app['analysis_mode_var'].set('Registradas (mesmo campo)')
    save = review.save_project
    def fail_new(data, path):
        if data['config'].clusters == 3:
            raise OSError('new checkpoint unavailable')
        return save(data, path)
    monkeypatch.setattr(review, 'save_project', fail_new)
    errors = []
    monkeypatch.setattr(app['messagebox'], 'showerror', lambda *a: errors.append(a))
    app['analyze']()
    assert errors
    assert app['session']['prepared'] is prepared
    assert scientific_state(prepared) == before
    assert app['session']['project_path'] == str(archive)
    assert archive.read_bytes() == previous_bytes


def test_group_noop_edits_do_not_consume_selection_undo_and_undo_marks_dirty(app, tmp_path):
    from test_riima_review import prepared_dataset
    prepared = prepared_dataset(tmp_path)
    app['session']['prepared'] = prepared
    app['setup_review']()
    app['select_groups']([0, 1])
    group = app['group_edit']('create_group', 'Novo')
    app['group_edit']('rename_group', group, 'Renomeado', presentation=True)
    app['select_groups']([0])
    app['group_edit']('rename_group', group, 'Renomeado', presentation=True)
    app['select_groups']([1])
    app['group_edit']('recolor_group', group, tuple(prepared['review'].palette[group]), presentation=True)
    app['session']['project_dirty'] = False
    app['undo_group_current']()
    assert app['session']['project_dirty']
    assert prepared['review'].group_name(group) == 'Novo'
    assert app['selected_workspace_groups']() == [group]
    app['undo_group_current']()
    assert group not in prepared['review'].groups
    assert app['selected_workspace_groups']() == [0, 1]


def test_legacy_metadata_does_not_overwrite_catalog_colors_or_select_empty_provenance(app, tmp_path):
    from test_riima_review import prepared_dataset
    prepared = prepared_dataset(tmp_path)
    state = prepared['review']
    created = state.create_group('Vazio')
    state.recolor_group(0, (1, 2, 3))
    prepared['interpretation'].create_family([0], 'Associação histórica')
    app['session']['prepared'] = prepared
    app['setup_review']()
    app['apply_interpretation']('rename_cluster', 0, 'Nome legado')
    np.testing.assert_array_equal(state.palette[0], [1, 2, 3])
    state.selected_cluster = created
    app['family_group_list'].selection_set(0)
    app['select_legacy_mask']()
    assert app['selected_workspace_groups']() == [0]
    assert state.selected_cluster == created


def test_compact_global_actions_and_four_primary_steps(app, tmp_path):
    from test_riima_review import prepared_dataset
    assert 'step_buttons' not in app and 'stepbar' not in app
    assert 'export_menu' in app, 'exports must stay accessible above the workspace'
    prepared = prepared_dataset(tmp_path, count=1)
    app['session']['prepared'] = prepared
    app['setup_review']()
    app['go_to_step']('organize')
    app['refresh_export_actions']()
    menu = app['export_menu']
    assert str(menu.entrycget(0, 'state')) == 'normal'
    prepared['review'].annotate(0, 0, [212], 1)
    app['refresh_export_actions']()
    assert str(menu.entrycget(0, 'state')) == 'disabled'
    assert str(menu.entrycget(1, 'state')) == 'normal'
    app['go_to_step']('results')
    assert app['workflow_var'].get() == 'workbench'
    assert all(button.cget('image') for button in app['review_tool_buttons'].values())
    assert all(button in app['tooltips'] for button in app['review_tool_buttons'].values())


def test_default_gui_save_and_group_autosave_write_only_portable_project(app, tmp_path):
    from test_riima_review import prepared_dataset
    from test_export_integrity import file_bytes, scientific_state
    prepared = prepared_dataset(tmp_path, count=1)
    app['session'].update(prepared=prepared, project_path=None)
    app['setup_review']()
    prepared['review'].annotate(0, 0, [212], 1)
    before = scientific_state(prepared)
    assert app['autosave_current']()
    output = Path(prepared['output'])
    assert sorted(file_bytes(output)) == ['projeto.riima']
    assert scientific_state(prepared) == before
    previous_project = (output / 'projeto.riima').read_bytes()
    app['apply_interpretation']('rename_cluster', 0, 'Live interpretation')
    if app['session']['autosave_after']:
        app['root'].after_cancel(app['session']['autosave_after'])
    app['save_interpretation_metadata']()
    assert (output / 'projeto.riima').read_bytes() == previous_project
    assert app['session']['project_dirty']
    assert app['save_project_current']()
    assert not app['session']['project_dirty']
    assert sorted(file_bytes(output)) == ['projeto.riima']
    restored = review.load_project(output / 'projeto.riima')
    assert restored['interpretation'].names[0] == 'Live interpretation'
    assert restored['review'].has_pending


def test_tooltips_identify_icon_actions_on_keyboard_focus(app):
    assert 'tooltips' in app, 'compact controls need keyboard-accessible descriptions'
    button = app['review_tool_buttons']['erase']
    button.event_generate('<FocusIn>')
    app['root'].update_idletasks()
    tip = app['tooltips'][button]
    assert tip['popup'] is not None and tip['popup'].winfo_exists()
    assert 'Apagar marcas' in tip['text'] and 'exemplos' in tip['text']
    button.event_generate('<FocusOut>')
    assert tip['popup'] is None


def test_status_indicator_reserves_space_without_moving_status_text(app):
    root = app['root']
    root.update_idletasks()
    assert 'operation_slot' in app, 'the progress indicator needs a stable layout slot'
    label = app['operation_status']
    before = (label.winfo_x(), label.winfo_y(), label.winfo_width(), label.winfo_height())
    observed = []
    root.after(15, lambda: observed.append((label.winfo_x(), label.winfo_y(),
                                           label.winfo_width(), label.winfo_height())))
    app['run_operation']('Processando', lambda: time.sleep(.08))
    root.update_idletasks()
    assert observed == [before]
    assert (label.winfo_x(), label.winfo_y(), label.winfo_width(), label.winfo_height()) == before
    assert not app['operation_progress'].winfo_ismapped()


def test_preview_title_does_not_push_zoom_controls_outside_small_window(app):
    root = app['root']
    previous_geometry, previous_title = root.geometry(), app['preview_title_var'].get()
    try:
        root.geometry('1080x700')
        app['preview_title_var'].set('1: campo_sintetico_com_nome_muito_extenso_da_lâmina.png | Sobreposição')
        root.update()
        controls = app['zoom_toolbar'].winfo_children()
        right = root.winfo_rootx() + root.winfo_width()
        for button in controls:
            assert button.winfo_ismapped()
            assert button.winfo_rootx() + button.winfo_reqwidth() <= right, button.cget('text')
    finally:
        app['preview_title_var'].set(previous_title)
        root.geometry(previous_geometry)


def test_analysis_mask_view_preserves_support_without_export_or_reclassification(app, tmp_path):
    from test_riima_review import prepared_dataset
    from test_export_integrity import file_bytes, scientific_state
    assert 'open_mask_comparison' not in app
    prepared = prepared_dataset(tmp_path, count=1)
    item = prepared['review'].items[0]
    raw = item['optical']
    raw['valid'] = raw['valid'].copy()
    raw['valid'][0, :4] = False
    item['valid'] = raw['valid'].copy()
    item['display_valid'] = raw['valid'].copy()
    app['session']['prepared'] = prepared
    app['setup_review']()
    before, state = file_bytes(tmp_path), scientific_state(prepared)
    app['view_type_var'].set('layers')
    app['on_view_type_change']()
    app['review_display_var'].set('Máscara')
    app['change_preview']()
    np.testing.assert_array_equal(app['session']['current_rgb'][..., 0], item['valid'].astype(np.uint8) * 255)
    assert int(raw['valid'].sum()) == 380 and raw['valid'].size == 384
    assert file_bytes(tmp_path) == before and scientific_state(prepared) == state


def test_historical_registered_mask_without_overlap_renders_saved_domain(app, tmp_path):
    from test_riima_review import prepared_dataset
    from test_export_integrity import scientific_state
    assert 'open_mask_comparison' not in app
    fixture = prepared_dataset(tmp_path, count=1)
    prepared = core.prepare_analysis([fixture['review'].items[0]['path']], str(tmp_path / 'registered'),
        optical_cfg=core.Config(clusters=2, feature_blur_sigma=0), mask_mode='overlap_only', publish_outputs=False)
    prepared['review'] = review.ClusterReview.from_prepared(prepared)
    prepared['optical'].pop('overlap', None)
    path = tmp_path / 'historical-mask.riima'
    review.save_project(prepared, path)
    loaded = review.load_project(path)
    before = scientific_state(loaded)
    for key in ('initial', 'labels', 'locked', 'valid', 'display_valid'):
        np.testing.assert_array_equal(loaded['review'].items[0][key], prepared['review'].items[0][key])
    app['show_loaded_project'](loaded, str(path))
    app['view_type_var'].set('layers')
    app['on_view_type_change']()
    app['review_display_var'].set('Máscara')
    app['change_preview']()
    assert loaded['optical']['mask_mode'] == 'overlap_only'
    assert 'overlap' not in prepared['optical'] and 'overlap' not in loaded['optical']
    assert loaded['review'].mode == 'registered'
    assert scientific_state(loaded) == before
    np.testing.assert_array_equal(app['session']['current_rgb'][..., 0],
                                  prepared['review'].items[0]['valid'].astype(np.uint8) * 255)


def test_context_similarity_matches_reference_and_caches_science_only(app, tmp_path, monkeypatch):
    import threading
    from riima import report_data
    from test_riima_review import prepared_dataset
    from test_export_integrity import file_bytes, scientific_state
    assert 'open_group_similarity' in app, 'group comparison must work inside interpretation'
    prepared = prepared_dataset(tmp_path, clusters=3)
    state = prepared['review']
    app['session'].update(prepared=prepared, workspace_selected_groups=[0])
    app['setup_review']()
    app['go_to_step']('organize')
    before, science = file_bytes(tmp_path), scientific_state(prepared)
    collect = report_data.collect_report_data
    calls = []
    def counted(*args, **kwargs):
        calls.append(threading.get_ident())
        return collect(*args, **kwargs)
    monkeypatch.setattr(report_data, 'collect_report_data', counted)
    window = app['open_group_similarity']()
    rows = app['session']['similarity_rows']
    reference = collect(prepared)
    selected = reference['active'].index(next(i for i,g in enumerate(reference['groups']) if g['id'] == 0))
    for row in rows:
        other = reference['active'].index(next(i for i,g in enumerate(reference['groups']) if g['id'] == row['id']))
        assert row['similarity'] == reference['similarity'][selected, other]
    assert [r['similarity'] for r in rows] == sorted([r['similarity'] for r in rows], reverse=True)
    assert calls and calls[0] != threading.get_ident()
    assert file_bytes(tmp_path) == before and scientific_state(prepared) == science
    window.destroy()
    app['apply_interpretation']('rename_cluster', 1, 'Renamed without recalculation')
    window = app['open_group_similarity']()
    assert len(calls) == 1
    assert any('Renamed without recalculation' in r['name'] for r in app['session']['similarity_rows'])
    window.destroy()
    state.exclude(0, [195], True)
    notices = []
    monkeypatch.setattr(app['messagebox'], 'showinfo', lambda *args: notices.append(args))
    assert app['open_group_similarity']() is None
    assert notices and len(calls) == 1
    app['refine_review_cluster']()
    window = app['open_group_similarity']()
    assert len(calls) == 2
    window.destroy()


def test_similarity_dialog_closes_on_edits_without_discarding_presentation_cache(app, tmp_path):
    from test_riima_review import prepared_dataset
    prepared = prepared_dataset(tmp_path, clusters=3)
    app['session']['prepared'] = prepared
    app['setup_review']()
    app['go_to_step']('organize')
    app['session']['workspace_selected_groups'] = [0]
    window = app['open_group_similarity']()
    cached = app['session']['similarity_cache']
    app['apply_interpretation']('rename_cluster', 1, 'New visible name')
    assert window.winfo_exists(), 'renaming updates the open ranking without recomputing science'
    assert app['session']['similarity_cache'] is cached
    window = app['open_group_similarity']()
    assert any('New visible name' in row['name'] for row in app['session']['similarity_rows'])
    app['apply_interpretation']('recolor_cluster', 1, (9, 136, 66))
    assert window.winfo_exists(), 'recoloring updates the open ranking without recomputing science'
    assert app['session']['similarity_cache'] is cached
    window = app['open_group_similarity']()
    app['apply_interpretation']('merge_clusters', [0, 1])
    assert not window.winfo_exists(), 'a nonmodal ranking must not retain obsolete group IDs'
    assert app['session']['similarity_cache'] is None


def test_disabled_listbox_events_cannot_change_selection_or_enable_review_during_worker(app, tmp_path):
    import threading
    from types import SimpleNamespace
    from test_riima_review import prepared_dataset
    from test_export_integrity import scientific_state
    prepared = prepared_dataset(tmp_path)
    state = prepared['review']
    app['session']['prepared'] = prepared
    app['setup_review']()
    app['go_to_step']('organize')
    box = app['optical_group_list']
    app['group_tree'].selection_set(())
    _select_group_rows(app, 0)
    app['on_group_selection']()
    state.annotate(0, state.selected_cluster, [212], 1)
    app['refresh_review_controls']()
    before = scientific_state(prepared)
    chosen, selected = list(app['session']['workspace_selected_groups']), _group_row_indices(app)
    release = threading.Event()
    observed = []
    def inspect():
        observed.append((_group_row_indices(app), list(app['session']['workspace_selected_groups']),
                         scientific_state(prepared), app['review_update_btn'].instate(['disabled']),
                         all(button.instate(['disabled']) for button in app['review_tool_buttons'].values())))
        release.set()
    def click_while_disabled():
        bounds = box.bbox(1)
        app['select_list_item'](SimpleNamespace(widget=box, y=bounds[1] + 2, state=0))
        box.event_generate('<<TreeviewSelect>>', when='tail')
        app['root'].after(15, inspect)
    app['root'].after(15, click_while_disabled)
    try:
        app['run_operation']('Checking disabled controls', lambda: release.wait(5))
    finally:
        release.set()
    assert observed == [(selected, chosen, before, True, True)]


def test_review_mutators_cannot_change_scientific_state_during_worker(app, tmp_path, monkeypatch):
    import threading
    from test_riima_review import prepared_dataset
    from test_export_integrity import scientific_state
    prepared = prepared_dataset(tmp_path)
    app['session']['prepared'] = prepared
    app['setup_review']()
    app['go_to_step']('organize')
    app['session']['workspace_selected_groups'] = [0]
    before = scientific_state(prepared)
    release = threading.Event()
    observations = []
    monkeypatch.setattr(app['messagebox'], 'showerror', lambda *args: None)
    monkeypatch.setattr(app['messagebox'], 'showinfo', lambda *args: None)
    def queued_callbacks():
        for name, args in [('selected_groups_background', (True,)), ('reset_review_group', ()),
                           ('apply_interpretation', ('rename_cluster', 0, 'Queued edit')),
                           ('approve_review_cluster', ()), ('reopen_review_cluster', ())]:
            app[name](*args)
            observations.append((name, scientific_state(prepared)))
        release.set()
    app['root'].after(15, queued_callbacks)
    try:
        app['run_operation']('Checking queued commands', lambda: release.wait(5))
    finally:
        release.set()
    assert observations
    assert all(snapshot == before for _, snapshot in observations), [name for name, state in observations if state != before]


@pytest.mark.parametrize('key', ['<Down>', '<Key-space>'])
def test_busy_treeview_keyboard_cannot_change_selection(app, tmp_path, key):
    import threading
    from test_riima_review import prepared_dataset
    prepared = prepared_dataset(tmp_path)
    app['session']['prepared'] = prepared
    app['setup_review']()
    app['go_to_step']('interpretation')
    tree = app['group_tree']
    app['select_groups']([0])
    tree.focus('1' if key == '<Key-space>' else '0')
    app['root'].focus_force()
    tree.focus_set()
    app['root'].update()
    observations, release = [], threading.Event()
    def key_while_busy():
        tree.event_generate(key)
        def inspect():
            observations.append((tree.selection(), app['selected_workspace_groups']()))
            release.set()
        app['root'].after(15, inspect)
    app['root'].after(15, key_while_busy)
    try:
        app['run_operation']('Keyboard during calculation', lambda: release.wait(5))
    finally:
        release.set()
    assert observations == [(('0',), [0])]


def test_similarity_dialog_may_close_while_worker_finishes(app, tmp_path, monkeypatch):
    import threading
    from riima import report_data
    from test_riima_review import prepared_dataset
    prepared = prepared_dataset(tmp_path)
    app['session']['prepared'] = prepared
    app['setup_review']()
    app['go_to_step']('organize')
    app['session']['workspace_selected_groups'] = [0]
    app['invalidate_similarity']()
    release = threading.Event()
    collect = report_data.collect_report_data
    def delayed(*args, **kwargs):
        assert release.wait(5)
        return collect(*args, **kwargs)
    monkeypatch.setattr(report_data, 'collect_report_data', delayed)
    errors = []
    monkeypatch.setattr(app['root'], 'report_callback_exception', lambda *args: errors.append(args))
    def close_dialog():
        windows = [w for w in app['root'].winfo_children()
                   if isinstance(w, tk.Toplevel) and w.title() == 'Similaridade entre grupos']
        assert len(windows) == 1
        windows[0].destroy()
        release.set()
    app['root'].after(20, close_dialog)
    try:
        assert app['open_group_similarity']() is None
        assert not errors and not app['session']['busy']
    finally:
        release.set()


def test_similarity_long_names_wrap_and_keyboard_selection_reveals_full_candidate(app, tmp_path):
    from tkinter import ttk, font as tkfont
    from test_riima_review import prepared_dataset
    prepared = prepared_dataset(tmp_path)
    selected_name = 'Grupo selecionado ' + 'caracterização petrográfica extensa ' * 5
    candidate_name = 'Grupo comparado. ' + 'interpretação detalhada da lâmina ' * 5
    prepared['interpretation'].names = {0: selected_name, 1: candidate_name}
    app['session']['prepared'] = prepared
    app['setup_review']()
    app['go_to_step']('organize')
    app['session']['workspace_selected_groups'] = [0]
    window = app['open_group_similarity']()
    window.geometry('480x400')
    app['root'].update()
    panel = window.winfo_children()[0]
    header = next(child for child in panel.winfo_children()
                  if isinstance(child, ttk.Label) and selected_name in child.cget('text'))
    assert tkfont.Font(font=header.cget('font') or 'TkDefaultFont').measure(selected_name) > window.winfo_width()
    assert header.winfo_reqwidth() <= window.winfo_width() - 20
    tree = next(child for child in panel.winfo_children() if isinstance(child, ttk.Treeview))
    tree.focus_force()
    tree.selection_set('1')
    tree.focus('1')
    tree.event_generate('<<TreeviewSelect>>')
    app['root'].update()
    assert any(candidate_name in text for text in _collect_label_texts(window))
    full_name = next(child for child in panel.winfo_children()
                     if isinstance(child, ttk.Label) and candidate_name in child.cget('text'))
    assert 'Afinidade' in full_name.cget('text')
    assert full_name.winfo_reqwidth() <= window.winfo_width() - 20
    window.destroy()


def test_review_radius_slider_has_integer_image_pixel_range(app, tmp_path):
    from tkinter import ttk
    from test_riima_review import prepared_dataset
    app['session']['prepared'] = prepared_dataset(tmp_path)
    app['setup_review']()
    app['go_to_step']('refine')
    app['root'].update()
    scale = app['review_radius_scale']
    assert isinstance(scale, ttk.Scale)
    assert (float(scale['from']), float(scale['to'])) == (1, 150)
    scale.set(149.7)
    assert app['review_radius_var'].get() == 150
    app['review_radius_var'].set(0)
    assert app['review_radius_var'].get() == 1
    app['review_radius_var'].set(151)
    assert app['review_radius_var'].get() == 150
    scale.focus_force()
    app['root'].update()
    scale.event_generate('<Left>')
    assert app['review_radius_var'].get() == 149


@pytest.mark.parametrize('zoom', [.5, 1, 4])
def test_brush_outline_tracks_radius_without_redrawing_pixels(app, tmp_path, zoom):
    from types import SimpleNamespace
    from test_riima_review import prepared_dataset
    app['session']['prepared'] = prepared_dataset(tmp_path)
    app['setup_review']()
    app['go_to_step']('refine')
    app['review_tool_var'].set('positive')
    app['review_radius_var'].set(7)
    app['session']['viewport'] = viewer.ViewportState(zoom=zoom, center_x=12, center_y=8)
    app['render_preview_canvas']()
    canvas = app['preview_canvas']
    raster = tuple(canvas.find_all())
    before = app['session']['current_rgb'].copy()
    app['on_canvas_motion'](SimpleNamespace(x=200, y=160))
    circles = canvas.find_withtag('brush_cursor')
    assert len(circles) == 2
    assert canvas.coords(circles[0]) == pytest.approx([200-7*zoom, 160-7*zoom, 200+7*zoom, 160+7*zoom])
    app['on_canvas_motion'](SimpleNamespace(x=210, y=170))
    assert canvas.find_withtag('brush_cursor') == circles
    assert all(item in canvas.find_all() for item in raster)
    np.testing.assert_array_equal(before, app['session']['current_rgb'])
    app['review_tool_var'].set('pan')
    assert not canvas.find_withtag('brush_cursor')
    app['review_tool_var'].set('restore')
    app['on_canvas_motion'](SimpleNamespace(x=200, y=160))
    assert canvas.find_withtag('brush_cursor')
    canvas.event_generate('<Leave>')
    assert not canvas.find_withtag('brush_cursor')
    app['session']['compare_mode'] = True
    app['on_canvas_motion'](SimpleNamespace(x=200, y=160))
    assert not canvas.find_withtag('brush_cursor')
    app['session']['compare_mode'] = False


@pytest.mark.parametrize('fail', [False, True])
def test_processing_keeps_tk_responsive_and_restores_controls(app, fail):
    import threading
    root = app['root']
    main_thread = threading.get_ident()
    observed = []
    ready = threading.Event()
    release = threading.Event()
    button = app['analyze_btn']
    previous = button.state()

    def heartbeat():
        if not ready.is_set():
            root.after(10, heartbeat)
            return
        observed.append((app['session']['busy'], button.instate(['disabled']),
                         app['operation_progress'].winfo_ismapped(), app['status_var'].get()))
        release.set()

    def work():
        assert threading.get_ident() != main_thread
        app['set_status']('Etapa de teste')
        ready.set()
        assert release.wait(5)
        if fail:
            raise ValueError('falha controlada')
        return 42

    root.after(30, heartbeat)
    if fail:
        with pytest.raises(ValueError, match='falha controlada'):
            app['run_operation']('Processando', work)
    else:
        assert app['run_operation']('Processando', work) == 42
    assert observed and observed[0][:3] == (True, True, 1)
    assert not app['session']['busy']
    assert button.state() == previous
    assert not app['operation_progress'].winfo_ismapped()


def test_processing_preserves_outer_export_transaction(app, tmp_path):
    from riima.exports import export_batch, staged_file
    target = tmp_path / 'result.txt'
    target.write_text('previous')
    with pytest.raises(ValueError, match='rollback'):
        with export_batch():
            app['run_operation']('Salvando', lambda: staged_file(target).write_text('new'))
            assert target.read_text() == 'previous'
            raise ValueError('rollback')
    assert target.read_text() == 'previous'


def test_processing_collects_tk_variable_cycles_on_the_ui_thread(app, monkeypatch):
    import gc
    import threading
    previous = gc.get_threshold()
    gc.set_threshold(2**30, 2**30, 2**30)
    variable = tk.IntVar(app['root'])
    target_name = variable._name
    cycle = [variable]
    cycle.append(cycle)
    finalizers = []
    finalize = tk.Variable.__del__

    def record_finalizer(value):
        if value._name == target_name:
            finalizers.append(threading.get_ident())
        finalize(value)

    monkeypatch.setattr(tk.Variable, '__del__', record_finalizer)
    del variable, cycle

    def allocations():
        gc.set_threshold(1, 1, 1)
        return len([[] for _ in range(200)])

    try:
        assert app['run_operation']('Processando', allocations) == 200
        gc.collect()
        assert finalizers == [threading.get_ident()]
    finally:
        gc.set_threshold(*previous)


@pytest.mark.parametrize('mode', ['direct', 'trained', 'unavailable'])
def test_review_feedback_and_generalization_through_native_gui(app, tmp_path, monkeypatch, mode):
    from test_riima_review import prepared_dataset
    prepared = prepared_dataset(tmp_path)
    state = prepared['review']
    target = int(state.items[0]['initial'][8, 3])
    state.selected_cluster = target
    state.annotate(0, target, [8*24+20], 1)
    if mode != 'direct':
        state.annotate(0, target, [8*24+3], -1)
    before = [item['labels'].copy() for item in state.items]
    initial = [item['initial'].copy() for item in state.items]
    errors = []
    monkeypatch.setattr(app['messagebox'], 'showerror', lambda *args: errors.append(args))
    if mode == 'unavailable':
        monkeypatch.delattr(cv2, 'ml')
    app['session']['prepared'] = prepared
    app['setup_review']()
    app['go_to_step']('refine')
    app['refine_review_cluster']()
    if mode == 'unavailable':
        assert errors and 'RTrees_create' in errors[0][1]
        assert state.has_pending
        assert not state.models
        for item, previous in zip(state.items, before):
            np.testing.assert_array_equal(item['labels'], previous)
    else:
        assert not errors and not state.has_pending
        if mode == 'direct':
            assert 'Suporte óptico local aplicado' in app['status_var'].get()
            assert target not in state.models
            assert state.items[1]['labels'][8, 23] == target  # Exact Lab/T match, never marked.
            assert np.all(state.items[1]['labels'][:, 21:] == target)
            assert np.all(state.items[1]['labels'][:, :14] == before[1][:, :14])
            assert state.items[0]['labels'][8, 20] == target
        else:
            assert 'Modelo treinado' in app['status_var'].get()
            assert target in state.models
            assert np.all(state.items[1]['labels'][:, 14:] == target)
            assert np.all(state.items[1]['labels'][:, :14] != target)
            assert state.items[0]['labels'][8, 20] == target
            assert state.items[0]['labels'][8, 3] != target
    for item, original in zip(state.items, initial):
        np.testing.assert_array_equal(item['initial'], original)
    assert not app['session']['busy']


def test_autosave_publication_keeps_event_loop_alive(app, tmp_path, monkeypatch):
    import threading
    from riima import exports
    from test_riima_review import prepared_dataset
    prepared = prepared_dataset(tmp_path)
    app['session'].update(prepared=prepared, project_path=str(tmp_path / 'project.riima'))
    app['setup_review']()
    app['go_to_step']('refine')
    ui_thread = threading.get_ident()
    original = exports._Export.publish
    publishing = threading.Event()
    heartbeat = threading.Event()
    observations = []

    def publish(operation):
        publishing.set()
        assert threading.get_ident() != ui_thread, 'Publication blocks the Tk thread'
        assert heartbeat.wait(5)
        return original(operation)

    def tick():
        if not publishing.is_set():
            app['root'].after(10, tick)
            return
        observations.append((app['session']['busy'], app['operation_progress'].winfo_ismapped()))
        heartbeat.set()

    monkeypatch.setattr(exports._Export, 'publish', publish)
    errors = []
    monkeypatch.setattr(app['messagebox'], 'showerror', lambda *args: errors.append(args))
    timer = app['root'].after(10, tick)
    try:
        assert app['autosave_current']()
        assert not errors
        assert observations == [(True, 1)]
        assert Path(app['session']['project_path']).is_file()
    finally:
        app['root'].after_cancel(timer)


@pytest.mark.parametrize('fail', [False, True])
def test_closing_diagnostics_during_processing_does_not_touch_destroyed_widgets(app, tmp_path, monkeypatch, fail):
    import threading
    from test_riima_review import prepared_dataset
    app['session']['prepared'] = prepared_dataset(tmp_path)
    app['setup_review']()
    root = app['root']
    release = threading.Event()
    errors = []
    monkeypatch.setattr(root, 'report_callback_exception', lambda *args: errors.append(args))
    evaluate = core.evaluate_cluster_counts

    def delayed(*args, **kwargs):
        assert release.wait(5)
        if fail:
            raise ValueError('diagnostic failure')
        return evaluate(*args, **kwargs)

    monkeypatch.setattr(core, 'evaluate_cluster_counts', delayed)
    previous = set(root.winfo_children())
    app['open_cluster_count_diagnostics']()
    window, = [w for w in root.winfo_children() if w not in previous and isinstance(w, tk.Toplevel)]

    def close_when_running():
        if not app['session']['busy']:
            root.after(10, close_when_running)
            return
        window.destroy()
        release.set()

    root.after(10, close_when_running)
    deadline = time.monotonic() + 6
    try:
        while not release.is_set() and time.monotonic() < deadline:
            root.update()
        assert release.is_set()
        assert not errors
        assert not app['session']['busy']
        assert not window.winfo_exists()
    finally:
        release.set()
        if window.winfo_exists():
            window.destroy()


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
            for name in ('import', 'configuration'):
                app['set_section_open'](name, True)
            root.update()
            for child in descendants(app['sidebar']):
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
    assert app['save_project_current']()
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
    assert app['workflow_var'].get() == 'workbench'
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
    menu = app['export_menu']
    assert menu.entrycget(0, 'label') == 'Relatório descritivo (HTML)'
    target = tmp_path / 'report.html'
    with monkeypatch.context() as cancel:
        cancel.setattr(app['filedialog'], 'asksaveasfilename', lambda **kw: '')
        cancel.setattr(riima_metrics, 'export_metrics', lambda *a, **kw: pytest.fail('calculation after cancellation'))
        menu.invoke(0)
    assert not target.exists()
    monkeypatch.setattr(app['filedialog'], 'asksaveasfilename', lambda **kw: str(target))
    monkeypatch.setattr(app['messagebox'], 'showerror', lambda title, text: pytest.fail(text))
    menu.invoke(0)
    html = target.read_text(encoding='utf-8')
    assert html.index('id="classificacao"') < html.index('id="tamanho"')
    assert app['workflow_var'].get() == 'workbench'
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
    app['root'].focus_force()
    app['preview_canvas'].focus_set()
    app['root'].update()
    assert app['root'].focus_get() is app['preview_canvas']
    click(0)
    assert app['root'].focus_get() is groups
    click(2)
    assert _group_row_indices(app) == (2,)
    click(0, 1)  # Shift adds a nonadjacent group without selecting the intervening one.
    assert _group_row_indices(app) == (0, 2)
    assert app['selected_workspace_groups']() == [0, 2]
    click(0, 1)
    assert _group_row_indices(app) == (2,)
    click(1, 4)  # Ctrl toggles another individual group.
    assert set(_group_row_indices(app)) == {1, 2}
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
    app['go_to_step']('preparation')
    app['clusters_var'].set(4)
    app['analyze_btn'].invoke()
    prepared = app['session']['prepared']
    assert prepared['optical']['sample_X'].shape[1] == 8
    assert app['workflow_var'].get() == 'workbench'
    app['go_to_step']('interpretation')
    groups = app['optical_group_list']
    app['group_tree'].selection_set(())
    _select_group_rows(app, 0)
    _select_group_rows(app, 3)
    app['on_group_selection']()
    app['merge_optical_selected']()
    state = prepared['review']
    assert state.groups == {0: {0, 3}, 1: {1}, 2: {2}}
    app['group_tree'].selection_set(())
    _select_group_rows(app, 0)
    app['on_group_selection']()
    monkeypatch.setattr(app['simpledialog'], 'askstring', lambda *args, **kw: 'Grupo <á & ç>')
    app['rename_selected_group'](groups, 'cluster')
    monkeypatch.setattr(app['colorchooser'], 'askcolor', lambda **kw: ((9, 136, 66), '#098842'))
    app['recolor_selected_group'](groups, 'cluster')
    app['go_to_step']('interpretation')
    app['group_tree'].selection_set(()); _select_group_rows(app, state.class_ids.index(0))
    app['select_review_cluster']()
    app['review_approve_btn'].invoke()
    item = state.items[0]
    raw = item['initial'].copy()
    frozen = item['locked'].copy()
    app['group_tree'].selection_set(()); _select_group_rows(app, state.class_ids.index(1))
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
    app['export_light_metrics']()  # CSVs are an explicit export, independent of portable HTML.
    with (Path(prepared['output']) / 'metrics' / 'group_summary.csv').open(encoding='utf-8-sig', newline='') as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        assert int(row['pixels']) == np.count_nonzero(item['labels'] == int(row['group']))
    assert [Path(path).read_bytes() for path in paths] == originals


@pytest.mark.parametrize('image_count', [1, 2])
@pytest.mark.parametrize('mode', ['registered', 'independent'])
def test_classification_stays_in_current_stage_for_k_evaluation(app, tmp_path, monkeypatch, image_count, mode):
    rgb = synthetic_registration_image()
    paths = write_rgb_images(tmp_path, [rgb, warp_test_image(rgb, dx=3, dy=-2)][:image_count])
    app['session']['prepared'] = None
    app['session']['images'] = [dict(path=path, name=Path(path).name) for path in paths]
    app['rebuild_image_rows']()
    app['reference_var'].set(0)
    app['clusters_var'].set(2)
    app['analysis_mode_var'].set('Independentes (campos distintos)' if mode == 'independent' else 'Registradas (mesmo campo)')
    app['out_var'].set(str(tmp_path / 'new-analysis'))
    app['go_to_step']('classify')
    monkeypatch.setattr(app['messagebox'], 'showerror', lambda title, text: pytest.fail(text))
    app['analyze']()
    assert app['workflow_var'].get() == 'workbench'
    assert app['groups_context'].winfo_manager() == 'pack'
    assert app['optical_group_frame'].winfo_manager() == 'pack'
    assert not app['section_open_vars']['configuration'].get()
    prepared = app['session']['prepared']
    assert prepared['optical']['sample_X'].shape[1] == (4 * image_count if mode == 'registered' else 4)
    assert prepared['review'].mode == mode
    assert len(prepared['review'].items) == (1 if mode == 'registered' else image_count)
    assert len(prepared['optical']['registered_images']) == (image_count if mode == 'registered' else 1)
    assert sorted(p.relative_to(prepared['output']).as_posix()
                  for p in Path(prepared['output']).rglob('*') if p.is_file()) == ['projeto.riima']
    if mode == 'independent' and image_count == 2:
        from test_export_integrity import scientific_state
        state = prepared['review']
        pixel = int(np.flatnonzero(state.items[0]['valid'])[0])
        state.annotate(0, state.selected_cluster, [pixel], 1)
        before = scientific_state(prepared)
        previous_path = app['session']['project_path']
        app['analyze']()
        assert app['session']['prepared'] is not prepared
        assert Path(app['session']['project_path']).name == 'projeto_2.riima'
        assert scientific_state(prepared) == before
        restored = review.load_project(previous_path)['review']
        assert restored.has_pending and restored.dirty == state.dirty
        assert restored.groups == state.groups and restored.completed == state.completed
        np.testing.assert_array_equal(restored.palette, state.palette)
        np.testing.assert_array_equal(restored.centers, state.centers)
        for saved, original in zip(restored.items, state.items):
            for field in ('rgb', 'initial', 'labels', 'locked', 'valid', 'display_valid'):
                np.testing.assert_array_equal(saved[field], original[field])
            np.testing.assert_array_equal(saved['optical']['confidence'], original['optical']['confidence'])
            assert saved['annotations'].keys() == original['annotations'].keys()
            for cluster in original['annotations']:
                np.testing.assert_array_equal(saved['annotations'][cluster], original['annotations'][cluster])
        assert sorted(p.name for p in Path(prepared['output']).iterdir()) == ['projeto.riima', 'projeto_2.riima']


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
    assert app['save_project_current']()
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
    assert app['session']['prepared'] is old
    assert app['session']['project_path'] == str(old_project)
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
    assert app['session']['images'][0]['path'] == str(tmp_path / 'new.png')
    app['analyze']()
    assert app['session']['prepared'] is old
    assert old['output'] == str(tmp_path / 'out')


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
    _select_group_rows(app, 0, 1)
    app['on_group_selection']()
    app['review_display_var'].set('Máscara selecionada')
    before = [item['labels'].copy() for item in prepared['review'].items]
    app['clear_group_selection']()
    assert app['selected_workspace_groups']() == []
    assert _group_row_indices(app) == ()
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
    app['group_tree'].selection_set(()); _select_group_rows(app, 0)
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
    app['group_tree'].selection_set(()); _select_group_rows(app, 0)
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
    app['analysis_mode_var'].set('Registradas (mesmo campo)')
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
    app['group_tree'].selection_set(()); _select_group_rows(app, 1)
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
    _select_group_rows(app, 0, 1)
    app['merge_optical_selected']()
    state = prepared['review']
    assert state.groups == {0: {0, 1}, 2: {2}}
    app['apply_interpretation']('rename_cluster', 0, 'Matriz')
    assert 'Matriz' in _group_texts(app)[0]
    app['group_tree'].selection_set(()); _select_group_rows(app, 1)
    app['select_review_cluster']()
    assert state.selected_cluster == 2
    state.annotate(0, 2, [212], 1)
    warnings = []
    monkeypatch.setattr(app['messagebox'], 'showinfo', lambda title, message: warnings.append(message))
    _select_group_rows(app, 0, 1)
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
    assert app['groups_context'].winfo_manager() == 'pack'
    assert app['section_frames']['import'].master.winfo_manager() == 'pack'
    assert app['family_group_frame'].winfo_manager() == ''
    assert app['layers_panel'].winfo_manager() == ''
    app['group_tree'].selection_set(())
    _select_group_rows(app, 0, 1)
    app['on_group_selection']()
    app['review_display_var'].set('Máscara selecionada')
    app['change_preview']()
    expected = viewer.workspace_mask(prepared, [0, 1])
    np.testing.assert_array_equal(app['session']['current_rgb'][..., 0], expected.astype(np.uint8) * 255)
    app['merge_optical_selected']()
    assert prepared['review'].groups == {0: {0, 1}, 2: {2}}
    app['go_to_step']('refine')
    assert prepared['review'].selected_cluster == 0
    assert set(app['session']['_compare_sources']) == {'Imagem original', 'Classificação atual', 'Máscara selecionada', 'Máscara'}
    app['go_to_step']('results')
    assert app['groups_context'].winfo_manager() == 'pack'
    assert app['section_frames']['import'].master.winfo_manager() == 'pack'


def test_workspace_resume_restores_step_image_and_mask_selection(app, tmp_path, monkeypatch):
    from test_riima_review import prepared_dataset
    prepared = prepared_dataset(tmp_path)
    app['session']['prepared'] = prepared
    app['session']['project_path'] = str(tmp_path / 'workspace.riima')
    app['setup_review']()
    app['refresh_group_listboxes']()
    app['go_to_step']('organize')
    app['group_tree'].selection_set(())
    _select_group_rows(app, 1)
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
    app['workflow_var'].set('preparation')
    app['review_display_var'].set('Imagem')
    app['open_project']()
    assert app['workflow_var'].get() == 'workbench'
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
        assert app['groups_context'].winfo_manager() == 'pack'
        assert app['review_frame'].winfo_manager() == 'pack'
    for item, initial in zip(state.items, before):
        np.testing.assert_array_equal(item['labels'], initial)


def test_comparator_resume_restores_union_mask_and_presentation_settings(app, tmp_path, monkeypatch):
    from test_riima_review import prepared_dataset
    prepared = prepared_dataset(tmp_path)
    app['session']['prepared'] = prepared
    app['setup_review']()
    app['refresh_group_listboxes']()
    app['go_to_step']('organize')
    _select_group_rows(app, 0, 1)
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
    app['group_tree'].selection_set(())
    app['on_group_selection']()
    app['review_display_var'].set('Máscara selecionada')
    app['change_preview']()
    assert app['session']['workspace_selected_groups'] == []
    assert not np.any(app['session']['current_rgb'])
    _select_group_rows(app, 0, 1)
    app['on_group_selection']()
    app['go_to_step']('refine')
    app['go_to_step']('results')
    assert app['selected_workspace_groups']() == [0, 1]
    assert app['session']['prepared']['review'].selected_cluster == 0


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
    _select_group_rows(app, 0, 1)
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
    controls = ('out_var', 'automatic_output_var', 'reference_var', 'clusters_var', 'context_visible_var',
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
    before_selection = _group_row_indices(app)
    before_frames = {key: frame.winfo_manager() for key, frame in app['section_frames'].items()}
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
        assert _group_row_indices(app) == before_selection
        assert {key: frame.winfo_manager() for key, frame in app['section_frames'].items()} == before_frames
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
    assert app['workflow_var'].get() == 'workbench'
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
    assert app['workflow_var'].get() == 'workbench'
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
    """One Tk interpreter for GUI and native desktop control tests.

    Reinitializing Tk after destroying a root can fail on Windows/CPython 3.13,
    even when either group of tests passes in isolation. Keep the workbench
    root alive for the module; desktop controls use disposable Toplevel hosts.
    Record real image element arguments during construction for state tests,
    always calling the original ttk implementation.
    """
    from tkinter import ttk
    captured = {}
    image_elements = {}
    create_element = ttk.Style.element_create

    def record_element(style, name, kind, *args, **options):
        if kind == 'image':
            image_elements[name] = args
        return create_element(style, name, kind, *args, **options)

    def hook(ns):
        captured.update(ns)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ttk.Style, 'element_create', record_element)
        riima_gui.gui(_test_hook=hook)
    captured['style']._image_elements = image_elements
    captured['root'].update()
    yield captured
    captured['root'].destroy()


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
    session['prepared'] = None
    session['project_path'] = None
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
    assert set(app['session']['_compare_sources']) == {'Imagem original', 'Classificação atual', 'Máscara selecionada', 'Máscara'}
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
    app['group_tree'].selection_set(())
    _select_group_rows(app, 1)
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
        assert app['groups_context'].winfo_manager() == 'pack'
        assert app['review_frame'].winfo_manager() == 'pack'
        assert dataclasses.asdict(app['session']['viewport']) == viewport
    assert app['diagnostics_slot'].master is app['section_frames']['configuration']
    assert app['optical_group_frame'].master is app['groups_context']
    assert app['review_frame'].master is app['groups_context']

def test_diagnostics_block_is_compact_not_a_tall_bordered_section(app):
    """The diagnostics entry fits within the classification sidebar."""
    from tkinter import ttk
    diagnostics_frame = app["diagnostics_frame"]
    assert not isinstance(diagnostics_frame, ttk.Labelframe)
    children = diagnostics_frame.winfo_children()
    buttons = [w for w in children if isinstance(w, ttk.Button)]
    assert [button.cget('text') for button in buttons] == ['Diagnóstico de K...']


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
    assert texts[2:] == ['Abrir projeto...', 'Salvar projeto', 'Exportar']

def test_manual_cluster_family_actions_preserve_raw_and_refresh_comparator(analyzed_app, monkeypatch):
    app = analyzed_app
    prepared = app["session"]["prepared"]
    raw = prepared["optical"]["labels"].copy()
    interpretation = prepared["interpretation"]
    monkeypatch.setattr(app["simpledialog"], "askstring", lambda *a, **kw: "Feldspato")
    app['group_tree'].selection_set(())
    _select_group_rows(app, 0)
    _select_group_rows(app, 2)
    app["create_family_selected"]()
    assert interpretation.families[0]["clusters"] == [0, 2]
    app['group_tree'].selection_set(())
    _select_group_rows(app, 0)
    _select_group_rows(app, 2)
    app["merge_optical_selected"]()
    assert {0, 2} in interpretation.groups
    assert np.array_equal(prepared["optical"]["labels"], raw)
    assert set(app["session"]["_compare_sources"]) == {"Imagem original", "Classificação atual", "Máscara selecionada", "Máscara"}
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
    app['group_tree'].selection_set(())
    _select_group_rows(app, 1)
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
    _select_group_rows(app, 0, 1)
    app["merge_optical_selected"]()
    assert interpretation.to_dict() == before
    assert "famílias diferentes" in messages[0]
    app["apply_interpretation"]("reset")


@pytest.mark.parametrize('failure', ['second_image', 'review', 'metrics', 'publication'])
def test_explicit_scientific_export_rolls_back_project_and_all_images(app, tmp_path, monkeypatch, failure):
    from test_riima_review import prepared_dataset
    from test_export_integrity import file_bytes, scientific_state
    from riima import exports
    prepared = prepared_dataset(tmp_path)
    session = app['session']
    path = str(tmp_path / 'session.riima')
    session.update(prepared=prepared, project_path=path)
    app['setup_review']()
    assert app['export_scientific_current']()
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
        elif failure == 'metrics':
            from riima import metrics
            original = metrics._write_csv
            def fail(path, *args, **kwargs):
                if path.name == 'attribute_statistics.csv':
                    path.write_text('incomplete sampled statistics', encoding='utf-8')
                    raise OSError('metrics failed')
                return original(path, *args, **kwargs)
            patch.setattr(metrics, '_write_csv', fail)
        else:
            original = exports.os.replace
            def fail(source, target):
                if '.riima-export' in str(source) and Path(target).name == 'statistics.csv':
                    raise OSError('publication failed')
                return original(source, target)
            patch.setattr(exports.os, 'replace', fail)
        assert app['export_scientific_current']() is False
    assert messages and session['project_path'] == path
    assert prepared['workspace'] is workspace
    assert scientific_state(prepared) == state
    assert file_bytes(tmp_path) == before
    assert app['export_scientific_current']()


def test_scientific_full_export_includes_all_metrics_files(app, tmp_path):
    from test_riima_review import prepared_dataset
    prepared = prepared_dataset(tmp_path)
    app['session'].update(prepared=prepared, project_path=None)
    app['setup_review']()
    assert app['export_scientific_current']()
    assert {path.name for path in (Path(prepared['output']) / 'metrics').glob('*')} == {
        'group_summary.csv', 'regions.csv', 'attribute_statistics.csv',
        'attribute_separation.csv', 'metrics_info.json'}


@pytest.mark.parametrize('clean_sign, evidence', [(1, 'positive_conflicting'), (-1, 'negative_conflicting')])
def test_review_status_reports_clean_support_and_conflicting_profile_abstention(app, tmp_path, clean_sign, evidence):
    from test_riima_review import prepared_dataset
    prepared = prepared_dataset(tmp_path)
    state = prepared['review']
    target = int(state.items[0]['initial'][8, 3])
    state.selected_cluster = target
    state.annotate(0, target, [8 * 24 + 3], 1)
    state.annotate(0, target, [8 * 24 + 4], -1)
    state.annotate(0, target, [8 * 24 + 20], clean_sign)
    app['session']['prepared'] = prepared
    app['setup_review']()
    app['go_to_step']('refine')
    app['refine_review_cluster']()
    assert state.refinement_evidence[target] == evidence
    text = app['status_var'].get()
    assert 'Suporte óptico local aplicado' in text
    assert 'contraditórios' in text and 'sem generalização' in text


def test_project_write_failure_keeps_previous_archive_and_scientific_state(app, tmp_path, monkeypatch):
    from test_riima_review import prepared_dataset
    from test_export_integrity import fail_writes, scientific_state
    prepared = prepared_dataset(tmp_path)
    app['session'].update(prepared=prepared, project_path=None)
    assert app['save_project_current']()
    target = Path(prepared['output']) / 'projeto.riima'
    before = target.read_bytes()
    prepared['interpretation'].rename_cluster(0, 'Updated metadata')
    state = scientific_state(prepared)
    errors = []
    monkeypatch.setattr(app['messagebox'], 'showerror', lambda *args: errors.append(args))
    with monkeypatch.context() as patch:
        fail_writes(patch, 'npy')
        patch.setattr(core, 'save_session_outputs', lambda *args: pytest.fail('full export'))
        assert app['save_project_current']() is False
    assert target.read_bytes() == before
    assert errors
    assert scientific_state(prepared) == state
    assert app['save_project_current']()
    assert target.read_bytes() != before


@pytest.mark.parametrize('failure', ['review', 'publication'])
def test_analysis_then_explicit_export_failure_keeps_previous_scientific_files(app, tmp_path, monkeypatch, failure):
    from test_riima_review import prepared_dataset
    from test_export_integrity import file_bytes
    prepared = prepared_dataset(tmp_path, count=1)
    path = prepared['review'].items[0]['path']
    session = app['session']
    session.update(prepared=prepared, project_path=str(tmp_path / 'checkpoint.riima'),
                   images=[dict(path=path, name=Path(path).name)])
    app['setup_review']()
    assert app['export_scientific_current']()
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
    assert not messages  # Quiet analysis does not attempt either scientific export.
    assert session['prepared'] is not prepared
    assert session['prepared']['config'].clusters == 3
    after_analysis = file_bytes(Path(prepared['output']))
    assert all(after_analysis[name] == value for name, value in before.items())
    assert app['export_scientific_current']() is False
    assert messages
    assert file_bytes(Path(prepared['output'])) == after_analysis
    assert 'exportação científica falhou' in app['status_var'].get().lower()


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
    result = app['export_scientific_current']()
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
    assert listbox.winfo_class() == 'Treeview'
    assert style.lookup('Treeview', 'background').lower() == theme.BG_PANEL.lower()
    assert style.lookup('Treeview', 'foreground').lower() == theme.TEXT_PRIMARY.lower()
    assert style.lookup('Treeview', 'foreground', ('selected',)).lower() == theme.TEXT_ON_ACCENT.lower()
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
        app['set_section_open']('legend', True)
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
        app['set_section_open']('legend', False)
        app['draw_legend']()



@pytest.mark.parametrize('mode', ['fit', 'manual'])
def test_single_workbench_focus_and_resize_preserve_all_scientific_interaction(app, tmp_path, mode):
    from test_riima_review import prepared_dataset
    from test_export_integrity import scientific_state
    assert 'context_paned' in app
    assert 'stepbar' not in app and 'step_buttons' not in app
    assert not any(isinstance(widget, __import__('tkinter').ttk.Notebook)
                   for widget in app['root'].winfo_children())
    prepared = prepared_dataset(tmp_path)
    app['session']['prepared'] = prepared
    app['setup_review']()
    app['change_preview']()
    app['select_groups']([0, 1])
    app['review_tool_var'].set('negative')
    state = prepared['review']
    state.annotate(0, state.selected_cluster, [212], 1)
    viewport = viewer.ViewportState(2.25, 9.5, 8.5, mode)
    app['session']['viewport'] = viewport
    before = dataclasses.asdict(viewport)
    science = scientific_state(prepared)
    canvas, tree = app['preview_canvas'], app['group_tree']
    app['root'].update()
    width = canvas.winfo_width()
    app['context_paned'].sash_place(0, 320, 0)
    app['root'].update()
    panel_width = app['sidebar_container'].winfo_width()
    app['toggle_context_panel']()
    app['root'].update()
    assert not app['sidebar_container'].winfo_ismapped()
    assert 'Restaurar painel' in app['tooltips'][app['focus_panel_btn']]['text']()
    assert canvas.winfo_width() > width
    app['toggle_context_panel']()
    app['root'].update()
    app['render_preview_canvas']()
    assert app['sidebar_container'].winfo_width() == panel_width
    assert 'Recolher painel' in app['tooltips'][app['focus_panel_btn']]['text']()
    assert app['preview_canvas'] is canvas and app['group_tree'] is tree
    assert app['session']['viewport'] is viewport
    assert dataclasses.asdict(viewport) == before
    assert app['selected_workspace_groups']() == [0, 1]
    assert app['review_tool_var'].get() == 'negative'
    assert scientific_state(prepared) == science
    assert app['optical_group_frame'].winfo_manager() == 'pack'
    assert app['review_frame'].winfo_manager() == 'pack'
    app['context_paned'].sash_place(0, 288, 0)


def test_workbench_sections_are_independent_and_keep_the_canvas(app, tmp_path):
    from test_riima_review import prepared_dataset
    assert set(app['section_frames']) == {'import', 'configuration', 'legend'}
    assert 'mask_combo' not in app and 'mask_mode_var' not in app
    assert 'open_mask_comparison' not in app
    prepared = prepared_dataset(tmp_path)
    app['show_loaded_project'](prepared, str(tmp_path / 'p.riima'))
    canvas = app['preview_canvas']
    assert not app['section_open_vars']['import'].get()
    assert not app['section_open_vars']['configuration'].get()
    for name in ('import', 'configuration', 'legend'):
        app['set_section_open'](name, True)
        app['root'].update()
        assert app['section_frames'][name].winfo_manager() == 'pack'
        assert app['preview_canvas'] is canvas
        assert app['optical_group_frame'].winfo_manager() == 'pack'
        assert app['review_frame'].winfo_manager() == 'pack'
        app['set_section_open'](name, False)
    assert 'Máscara' in app['review_display_combo']['values']


@pytest.mark.parametrize('mode', ['registered', 'independent'])
def test_new_workbench_analysis_always_uses_auto_mask(app, tmp_path, monkeypatch, mode):
    from test_riima_review import prepared_dataset
    prepared = prepared_dataset(tmp_path)
    paths = [prepared['review'].items[0]['path']]
    app['session']['prepared'] = None
    app['session']['images'] = [dict(path=path, name=Path(path).name) for path in paths]
    app['rebuild_image_rows']()
    app['out_var'].set(str(tmp_path / 'new-auto'))
    app['analysis_mode_var'].set('Independentes (campos distintos)' if mode == 'independent'
                                 else 'Registradas (mesmo campo)')
    captured = []
    def stop_after_mask(*args, **kwargs):
        captured.append(kwargs['mask_mode'])
        raise ValueError('Expected stop after checking mask contract')
    monkeypatch.setattr(core if mode == 'registered' else review,
                        'prepare_analysis' if mode == 'registered' else 'prepare_independent_analysis',
                        stop_after_mask)
    messages = []
    monkeypatch.setattr(app['messagebox'], 'showerror', lambda *args: messages.append(args))
    app['analyze']()
    assert captured == ['auto']
    assert len(messages) == 1


@pytest.mark.parametrize('mode', ['layers', 'compare'])
@pytest.mark.parametrize('scaling', [1.0, 1.75, 2.0])
def test_resizing_context_panel_keeps_view_controls_inside_canvas_workspace(app, tmp_path, mode, scaling):
    from test_riima_review import prepared_dataset
    app['show_loaded_project'](prepared_dataset(tmp_path), str(tmp_path / 'view.riima'))
    root = app['root']
    previous = root.geometry()
    previous_scaling = float(root.tk.call('tk', 'scaling'))
    from riima import theme
    try:
        root.tk.call('tk', 'scaling', scaling)
        theme.apply(root, app['style'])
        root.geometry('1080x700')
        app['view_type_var'].set(mode)
        app['on_view_type_change']()
        root.update()
        viewport = app['session']['viewport']
        viewport_values = dataclasses.asdict(viewport)
        canvas = app['preview_canvas']
        app['context_paned'].sash_place(0, 850, 0)
        root.update()
        workspace = app['workspace']
        frame = app['compare_panel'] if mode == 'compare' else app['display_controls']
        left = workspace.winfo_rootx()
        right = left + workspace.winfo_width()
        def descendants(widget):
            for child in widget.winfo_children():
                yield child
                yield from descendants(child)
        for controls in (frame, app['view_type_row'], app['zoom_toolbar']):
            for control in descendants(controls):
                assert control.winfo_ismapped(), (scaling, mode, control)
                assert control.winfo_width() >= control.winfo_reqwidth(), (scaling, mode, control)
                assert left <= control.winfo_rootx()
                assert control.winfo_rootx() + control.winfo_width() <= right
        if mode == 'compare':
            comparison_widgets = app['compare_panel'].winfo_children()
            for label, combo in ((comparison_widgets[0], app['split_left_combo']),
                                 (comparison_widgets[2], app['split_right_combo'])):
                assert label.winfo_rootx() + label.winfo_width() <= combo.winfo_rootx()
                label_center = label.winfo_rooty() + label.winfo_height() / 2
                combo_center = combo.winfo_rooty() + combo.winfo_height() / 2
                assert abs(label_center - combo_center) <= 1
        title = app['preview_title_label']
        assert title.winfo_ismapped()
        assert title.winfo_width() >= title.winfo_reqwidth()
        assert title.winfo_height() >= title.winfo_reqheight()
        assert left <= title.winfo_rootx()
        assert title.winfo_rootx() + title.winfo_width() <= right
        assert title.winfo_rooty() >= workspace.winfo_rooty()
        assert title.winfo_rooty() + title.winfo_height() <= workspace.winfo_rooty() + workspace.winfo_height()
        assert canvas.winfo_width() >= 450
        assert app['preview_canvas'] is canvas and app['session']['viewport'] is viewport
        assert dataclasses.asdict(viewport) == viewport_values
    finally:
        app['context_paned'].sash_place(0, 288, 0)
        root.tk.call('tk', 'scaling', previous_scaling)
        theme.apply(root, app['style'])
        root.geometry(previous)


def test_group_tree_tooltip_tracks_full_hovered_and_focused_names(app, tmp_path):
    from types import SimpleNamespace
    from test_riima_review import prepared_dataset
    prepared = prepared_dataset(tmp_path)
    state = prepared['review']
    name = 'Mineral com textura e nome interpretativo muito extenso ' * 7
    state.rename_group(1, name)
    app['show_loaded_project'](prepared, str(tmp_path / 'names.riima'))
    tree = app['group_tree']
    app['sidebar_canvas'].yview_moveto(0)
    app['root'].update()
    tree.see('1')
    bbox = tree.bbox('1')
    app['on_group_hover'](SimpleNamespace(y=bbox[1] + bbox[3] // 2))
    assert name in app['tooltips'][tree]['text']()
    app['session']['hover_group'] = None
    tree.focus('1')
    assert name in app['tooltips'][tree]['text']()
    app['tooltips'][tree]['hide']()



@pytest.mark.parametrize('scaling', [1.75, 2.0])
@pytest.mark.parametrize('analyzed', [False, True])
def test_minimum_context_width_wraps_explanatory_labels_inside_the_panel(app, tmp_path, scaling, analyzed):
    from tkinter import ttk
    from riima import theme
    from test_riima_review import prepared_dataset
    root = app['root']
    previous_scaling = float(root.tk.call('tk', 'scaling'))
    previous_geometry = root.geometry()
    if analyzed:
        app['show_loaded_project'](prepared_dataset(tmp_path), str(tmp_path / 'min.riima'))
    else:
        app['session']['prepared'] = None
        app['change_preview']()
    def descendants(widget):
        for child in widget.winfo_children():
            yield child
            yield from descendants(child)
    try:
        root.tk.call('tk', 'scaling', scaling)
        theme.apply(root, app['style'])
        root.geometry('1080x700')
        app['set_section_open']('import', True)
        app['set_section_open']('configuration', True)
        app['context_paned'].sash_place(0, 252, 0)
        root.update()
        hint = next(widget for widget in app['sidebar'].winfo_children()
                    if isinstance(widget, ttk.Label)
                    and str(widget.cget('textvariable')) == str(app['workflow_hint_var']))
        assert hint.winfo_ismapped()
        padding = tuple(root.winfo_pixels(str(value)) for value in app['sidebar'].cget('padding'))
        usable_width = app['sidebar'].winfo_width() - padding[0] - padding[2]
        assert hint.winfo_reqwidth() <= usable_width
        assert hint.winfo_width() <= usable_width
        assert hint.winfo_height() >= hint.winfo_reqheight()
        assert ('Importe imagens' if not analyzed else 'Selecione para inspecionar') in app['workflow_hint_var'].get()
        labels = [widget for widget in descendants(app['sidebar'])
                  if isinstance(widget, ttk.Label) and int(widget.cget('wraplength') or 0) > 0]
        assert labels
        for label in labels:
            if label.winfo_ismapped():
                assert label.winfo_reqwidth() <= label.master.winfo_width(), label.cget('text')
    finally:
        root.tk.call('tk', 'scaling', previous_scaling)
        theme.apply(root, app['style'])
        app['context_paned'].sash_place(0, 288, 0)
        app['set_section_open']('import', False)
        app['set_section_open']('configuration', False)
        root.geometry(previous_geometry)


@pytest.mark.parametrize('case', ['empty', 'single', 'multiple', 'pending', 'approved', 'background'])
def test_selection_actions_follow_inspection_selection_and_group_protection(app, tmp_path, case):
    from test_riima_review import prepared_dataset
    from test_export_integrity import scientific_state
    prepared = prepared_dataset(tmp_path)
    app['show_loaded_project'](prepared, str(tmp_path / 'actions.riima'))
    state = prepared['review']
    if case == 'pending':
        state.annotate(0, 0, [212], 1)
    elif case == 'approved':
        state.approve(0)
    elif case == 'background':
        state.set_background([0], True)
    chosen = [] if case == 'empty' else [0, 1] if case == 'multiple' else [0]
    before = scientific_state(prepared)
    app['select_groups'](chosen)
    app['refresh_review_controls']()
    background = {button.cget('text'): button for button in app['background_actions'].winfo_children()}
    send = background['Enviar seleção ao fundo']
    restore = background['Restaurar seleção do fundo']
    assert app['clear_selection_btn'].instate(['disabled']) == (case == 'empty')
    assert app['activate_group_btn'].instate(['disabled']) == (case == 'empty')
    assert app['similarity_btn'].instate(['disabled']) == (case in ('empty', 'pending', 'background'))
    assert send.instate(['disabled']) == (case in ('empty', 'approved', 'background'))
    assert restore.instate(['disabled']) == (case != 'background')
    assert state.selected_cluster == 0
    assert scientific_state(prepared) == before
    if case == 'empty':
        assert app['review_tool_buttons']['positive'].instate(['!disabled'])
        assert app['review_tool_buttons']['negative'].instate(['!disabled'])


def test_selection_action_buttons_remain_disabled_during_processing(app, tmp_path):
    from test_riima_review import prepared_dataset
    from test_export_integrity import scientific_state
    app['show_loaded_project'](prepared_dataset(tmp_path), str(tmp_path / 'busy-actions.riima'))
    app['select_groups']([0, 1])
    before = scientific_state(app['session']['prepared'])
    observed = []
    def inspect():
        buttons = app['selection_actions'].winfo_children()
        observed.append(all(button.instate(['disabled']) for button in buttons))
        app['clear_group_selection']()
        app['selected_groups_background'](True)
        app['select_groups']([])
    app['root'].after(10, inspect)
    app['run_operation']('Processando', lambda: time.sleep(.05))
    assert observed == [True]
    assert app['selected_workspace_groups']() == [0, 1]
    assert scientific_state(app['session']['prepared']) == before
    assert app['clear_selection_btn'].instate(['!disabled'])


@pytest.mark.parametrize('embedded', [False, True])
def test_image_rows_expose_full_name_source_and_updated_status_to_keyboard_tooltips(app, tmp_path, embedded):
    from tkinter import ttk
    from test_riima_review import prepared_dataset
    from test_export_integrity import scientific_state
    prepared = prepared_dataset(tmp_path)
    app['show_loaded_project'](prepared, str(tmp_path / 'sources.riima'))
    name = 'campo_sintetico_com_nome_extenso_para_inspecao.png'
    relative = str(Path('imagens de referência') / name)
    other_name = 'outra_micrografia_com_nome_e_origem_distintos.png'
    other_relative = str(Path('outra pasta de imagens') / other_name)
    prepared['review'].items[1]['name'] = other_name
    prepared['review'].items[1]['path'] = other_relative
    prepared['optical']['image_names'].append(other_name)
    prepared['review'].items[0]['name'] = name
    prepared['review'].items[0]['path'] = relative
    prepared['optical']['image_names'][0] = name
    app['session']['images'] = [] if embedded else [dict(name=name, path=relative),
                                                     dict(name=other_name, path=other_relative)]
    before = scientific_state(prepared)
    app['rebuild_image_rows']()
    def descendants(widget):
        for child in widget.winfo_children():
            yield child
            yield from descendants(child)
    label = next(widget for widget in descendants(app['images_list_frame'])
                 if isinstance(widget, ttk.Label) and widget.cget('text') == name)
    assert label in app['tooltips']
    text = app['tooltips'][label]['text']
    assert name in (text() if callable(text) else text)
    assert str(Path(relative).resolve()) in (text() if callable(text) else text)
    assert str(label.cget('takefocus')) == '1'
    other = next(widget for widget in descendants(app['images_list_frame'])
                 if isinstance(widget, ttk.Label) and widget.cget('text') == other_name)
    other_text = app['tooltips'][other]['text']
    other_text = other_text() if callable(other_text) else other_text
    assert other_name in other_text and str(Path(other_relative).resolve()) in other_text
    assert str(Path(relative).resolve()) not in other_text
    app['set_section_open']('import', True)
    label.focus_force()
    app['root'].update()
    assert app['tooltips'][label]['popup'] is not None
    app['tooltips'][label]['hide']()
    if not embedded:
        app['update_row_statuses']({name: 'Confira o alinhamento: coordenadas preservadas; registro não verificado.'})
        app['update_row_statuses']({name: 'Registro de referência preservado.', other_name: 'Outro alinhamento a conferir.'})
        other_status = app['session']['_status_labels'][other_name]
        other_tip = app['tooltips'][other_status]['text']()
        assert other_status.cget('text') in other_tip and str(Path(other_relative).resolve()) in other_tip
        assert str(Path(relative).resolve()) not in other_tip
        status = app['session']['_status_labels'][name]
        assert status in app['tooltips']
        text = app['tooltips'][status]['text']
        text = text() if callable(text) else text
        assert status.cget('text') in text and str(Path(relative).resolve()) in text
    assert scientific_state(prepared) == before
    app['set_section_open']('import', False)


@pytest.mark.parametrize('panel_width', [288, 360])
def test_review_tool_buttons_keep_consistent_compact_width_when_panel_resizes(app, tmp_path, panel_width):
    from test_riima_review import prepared_dataset
    app['show_loaded_project'](prepared_dataset(tmp_path), str(tmp_path / 'compact.riima'))
    app['context_paned'].sash_place(0, panel_width, 0)
    app['root'].update()
    try:
        buttons = list(app['review_tool_buttons'].values())
        assert all(button.winfo_ismapped() for button in buttons)
        widths = [button.winfo_width() for button in buttons]
        assert max(widths) - min(widths) <= 1
        assert all(button.winfo_width() <= button.winfo_reqwidth() + 1 for button in buttons)
    finally:
        app['context_paned'].sash_place(0, 288, 0)


@pytest.mark.parametrize('operation', ['create', 'rename', 'delete', 'undo'])
def test_catalog_operations_finish_with_unsaved_status_and_preserve_expected_arrays(app, tmp_path, operation):
    from test_riima_review import prepared_dataset
    prepared = prepared_dataset(tmp_path)
    state = prepared['review']
    if operation == 'delete':
        empty = state.create_group('Grupo vazio')
    app['show_loaded_project'](prepared, str(tmp_path / 'catalog.riima'))
    original = [{key: item[key].copy() for key in ('initial', 'labels', 'locked', 'valid')}
                for item in state.items]
    initial_ids = set(state.groups)
    app['status_var'].set('Atualizando os grupos...')
    if operation == 'create':
        created = app['group_edit']('create_group', 'Novo grupo')
        assert set(state.groups) == initial_ids | {created} and not state.groups[created]
    elif operation == 'rename':
        app['group_edit']('rename_group', 0, 'Matriz atual', presentation=True)
        assert state.group_name(0) == 'Matriz atual'
    elif operation == 'delete':
        app['group_edit']('delete_group', empty)
        assert set(state.groups) == initial_ids - {empty}
    else:
        created = app['group_edit']('create_group', 'Temporário')
        app['status_var'].set('Atualizando os grupos...')
        app['undo_group_current']()
        assert set(state.groups) == initial_ids and created not in state.groups
    assert not app['session']['busy'] and app['session']['project_dirty']
    assert 'não salvas' in app['status_var'].get()
    assert not app['status_var'].get().endswith('...')
    for item, expected in zip(state.items, original):
        for key, value in expected.items():
            np.testing.assert_array_equal(item[key], value)


def test_catalog_failure_keeps_error_status_and_scientific_state(app, tmp_path, monkeypatch):
    from test_riima_review import prepared_dataset
    from test_export_integrity import scientific_state
    prepared = prepared_dataset(tmp_path)
    prepared['review'].approve(0)
    app['show_loaded_project'](prepared, str(tmp_path / 'protected.riima'))
    before = scientific_state(prepared)
    errors = []
    monkeypatch.setattr(app['messagebox'], 'showinfo', lambda *args: errors.append(args))
    app['group_edit']('delete_group', 0)
    assert errors and not app['session']['busy']
    assert scientific_state(prepared) == before
    assert not app['session']['project_dirty']
    assert 'Não foi possível' in app['status_var'].get()


@pytest.mark.parametrize('scaling', [1.75, 2.0])
def test_output_folder_controls_fit_minimum_panel_at_large_scaling_and_keep_behavior(app, tmp_path, monkeypatch, scaling):
    from tkinter import ttk
    from riima import theme
    from test_riima_review import prepared_dataset
    prepared = prepared_dataset(tmp_path)
    app['show_loaded_project'](prepared, str(tmp_path / 'output.riima'))
    root = app['root']
    previous_scaling = float(root.tk.call('tk', 'scaling'))
    previous_geometry = root.geometry()
    original = [{key: item[key].copy() for key in ('initial', 'labels', 'locked', 'valid')}
                for item in prepared['review'].items]
    try:
        root.tk.call('tk', 'scaling', scaling)
        theme.apply(root, app['style'])
        root.geometry('1080x700')
        app['set_section_open']('import', True)
        app['context_paned'].sash_place(0, 252, 0)
        root.update()
        button = next(widget for widget in app['outrow'].winfo_children() if isinstance(widget, ttk.Button))
        entry = next(widget for widget in app['outrow'].winfo_children() if isinstance(widget, ttk.Entry))
        check = next(widget for widget in app['images_step'].winfo_children() if isinstance(widget, ttk.Checkbutton))
        assert button.winfo_ismapped() and button.winfo_width() >= button.winfo_reqwidth()
        assert entry.winfo_ismapped() and entry.winfo_width() >= 20
        assert button.winfo_x() + button.winfo_width() <= app['outrow'].winfo_width()
        assert check.cget('text') == 'Pasta automática'
        assert check.winfo_reqwidth() <= check.master.winfo_width()
        assert button in app['tooltips'] and check in app['tooltips']
        assert 'novas imagens' in app['tooltips'][check]['text']
        app['automatic_output_var'].set(True)
        check.invoke()
        assert app['automatic_output_var'].get() is False
        check.invoke()
        assert app['automatic_output_var'].get() is True
        selected = str(tmp_path / 'destino escolhido')
        monkeypatch.setattr(app['filedialog'], 'askdirectory', lambda **kwargs: selected)
        button.invoke()
        assert app['out_var'].get() == selected and prepared['output'] == selected
        assert prepared['output_pending'] is False
        assert app['automatic_output_var'].get() is False
        monkeypatch.setattr(app['filedialog'], 'askdirectory', lambda **kwargs: '')
        button.invoke()
        assert app['out_var'].get() == selected and prepared['output'] == selected
        assert app['automatic_output_var'].get() is False
        for item, expected in zip(prepared['review'].items, original):
            for key, value in expected.items():
                np.testing.assert_array_equal(item[key], value)
    finally:
        root.tk.call('tk', 'scaling', previous_scaling)
        theme.apply(root, app['style'])
        app['context_paned'].sash_place(0, 288, 0)
        app['set_section_open']('import', False)
        root.geometry(previous_geometry)


@pytest.fixture
def desktop_style(app):
    """Isolate each control case in a Toplevel on the existing GUI interpreter."""
    host = tk.Toplevel(app['root'])
    host.withdraw()
    host.geometry('480x300+30+30')
    assert host.tk is app['root'].tk
    try:
        yield host, app['style']
    finally:
        host.destroy()
        app['root'].update_idletasks()


@pytest.mark.parametrize('name', [
    'TButton', 'Icon.TButton', 'Selected.TButton', 'Primary.TButton',
    'Toolbar.TMenubutton', 'TEntry', 'TCombobox', 'TSpinbox',
    'TCheckbutton', 'TRadiobutton', 'Treeview', 'Treeview.Heading',
    'Horizontal.TScale', 'Vertical.TScrollbar', 'TLabelframe',
])
def test_desktop_layouts_replace_platform_frames_and_indicators(desktop_style, name):
    _root, style = desktop_style
    def elements(layout):
        for element, options in layout:
            yield element
            yield from elements(options.get('children', []))
    names = list(elements(style.layout(name)))
    assert any(element.startswith('RIIMA.') for element in names), (name, names)
    assert not any(element.endswith(('.border', '.focus')) for element in names), (name, names)
    assert str(style.lookup(name, 'relief', default='flat')) == 'flat'


def test_flat_controls_preserve_pointer_keyboard_and_popup_behavior(desktop_style):
    from test_visual_desktop import _contrast
    import tkinter as tk
    from tkinter import ttk
    root, style = desktop_style
    root.geometry('480x300+30+30')
    selected = tk.BooleanVar(root, False)
    check = ttk.Checkbutton(root, text='Opção', variable=selected)
    check.pack()
    value = tk.StringVar(root, '3')
    spin = ttk.Spinbox(root, from_=1, to=9, textvariable=value)
    spin.pack()
    combo = ttk.Combobox(root, values=('Primeiro', 'Segundo'), state='readonly')
    combo.current(0)
    combo.pack()
    entry = ttk.Entry(root)
    entry.pack()
    root.deiconify()
    root.update()
    check.focus_force()
    check.event_generate('<space>')
    root.update()
    assert selected.get()
    check.state(['disabled'])
    check.event_generate('<space>')
    assert selected.get()
    spin.focus_force()
    spin.event_generate('<Up>')
    root.update()
    assert value.get() == '4'
    points = [(x, y) for x in range(spin.winfo_width() - 24, spin.winfo_width())
              for y in range(spin.winfo_height()) if spin.identify(x, y).endswith('uparrow')]
    assert points, 'Spinbox increment must remain a pointer target'
    x, y = points[len(points) // 2]
    spin.event_generate('<ButtonPress-1>', x=x, y=y)
    spin.event_generate('<ButtonRelease-1>', x=x, y=y)
    root.update()
    assert value.get() == '5'
    combo.focus_force()
    combo.event_generate('<Down>')
    root.update()
    popup = root.tk.call('ttk::combobox::PopdownWindow', str(combo))
    assert root.tk.getboolean(root.tk.call('winfo', 'ismapped', popup))
    root.tk.call('ttk::combobox::Unpost', str(combo))
    entry.focus_force()
    root.update()
    assert root.focus_get() is entry
    entry.insert(0, 'Texto editável')
    assert entry.get() == 'Texto editável'
    assert _contrast(style.lookup('TEntry', 'foreground'),
                     style.lookup('TEntry', 'fieldbackground')) >= 4.5


def test_flat_theme_reapplication_keeps_live_elements_and_compact_targets(desktop_style):
    from riima import theme
    from tkinter import ttk
    root, style = desktop_style
    button = ttk.Button(root, text='Ação', style='Icon.TButton', image='')
    button.pack()
    root.update_idletasks()
    original = (button.winfo_reqwidth(), button.winfo_reqheight())
    image_names = set(root.tk.call('image', 'names'))
    theme.apply(root._root(), style)
    root.update_idletasks()
    assert set(root.tk.call('image', 'names')) == image_names
    assert (button.winfo_reqwidth(), button.winfo_reqheight()) == original
    assert original[0] >= 28 and original[1] >= 28
    assert int(style.lookup('TButton', 'width')) == 0


def test_editable_controls_share_compact_height(desktop_style):
    from tkinter import ttk
    root, _style = desktop_style
    widgets = [ttk.Entry(root), ttk.Combobox(root), ttk.Spinbox(root, from_=1, to=10)]
    for widget in widgets:
        widget.pack()
    root.update_idletasks()
    heights = [widget.winfo_reqheight() for widget in widgets]
    assert max(heights) - min(heights) <= 4, heights


def test_flat_scrollbar_preserves_arrow_steps_and_thumb_targets(desktop_style):
    from tkinter import ttk
    root, _style = desktop_style
    frame = ttk.Frame(root)
    frame.pack(fill='both', expand=True)
    tree = ttk.Treeview(frame, height=4)
    tree.pack(side='left', fill='both', expand=True)
    scroll = ttk.Scrollbar(frame, command=tree.yview)
    scroll.pack(side='right', fill='y')
    tree.configure(yscrollcommand=scroll.set)
    for i in range(100):
        tree.insert('', 'end', text=str(i))
    root.deiconify()
    root.update()
    before = tree.yview()
    targets = {}
    for y in range(scroll.winfo_height()):
        name = scroll.identify(scroll.winfo_width() // 2, y)
        for suffix in ('uparrow', 'downarrow', 'thumb'):
            if name.endswith(suffix):
                targets[suffix] = y
    assert set(targets) == {'uparrow', 'downarrow', 'thumb'}
    x, y = scroll.winfo_width() // 2, targets['downarrow']
    scroll.event_generate('<ButtonPress-1>', x=x, y=y)
    scroll.event_generate('<ButtonRelease-1>', x=x, y=y)
    root.update()
    assert tree.yview()[0] > before[0]
    previous = tree.yview()[0]
    root.update()
    ys = [y for y in range(scroll.winfo_height())
          if scroll.identify(x, y).endswith('thumb')]
    y = ys[len(ys) // 2]
    scroll.event_generate('<ButtonPress-1>', x=x, y=y)
    scroll.event_generate('<B1-Motion>', x=x, y=y+20)
    scroll.event_generate('<ButtonRelease-1>', x=x, y=y+20)
    root.update()
    assert tree.yview()[0] > previous


def test_flat_scale_keeps_keyboard_and_pointer_drag(desktop_style):
    from tkinter import ttk
    root, _style = desktop_style
    scale = ttk.Scale(root, from_=0, to=100, length=240)
    scale.pack()
    root.deiconify()
    root.update()
    scale.set(10)
    scale.focus_force()
    scale.event_generate('<Right>')
    root.update()
    assert scale.get() > 10
    before = scale.get()
    y = scale.winfo_height() // 2
    xs = [x for x in range(scale.winfo_width()) if scale.identify(x, y).endswith('slider')]
    assert xs
    x = xs[len(xs) // 2]
    scale.event_generate('<ButtonPress-1>', x=x, y=y)
    scale.event_generate('<B1-Motion>', x=x+50, y=y)
    scale.event_generate('<ButtonRelease-1>', x=x+50, y=y)
    root.update()
    assert scale.get() > before


@pytest.mark.parametrize('name', ['button', 'primary', 'option'])
def test_focused_buttons_preserve_hover_press_feedback_and_focus_ring(desktop_style, name):
    _root, style = desktop_style
    definition = style._image_elements['RIIMA.' + name]
    def image_for(*states):
        for *required, image in definition[1:]:
            if all(state in states for state in required):
                return image
        return definition[0]
    def pixel(image, x, y):
        return image.tk.call(str(image), 'get', x, y)
    focused = image_for('focus')
    for state in ('active', 'pressed'):
        interactive = image_for('focus', state)
        assert pixel(interactive, 12, 12) != pixel(focused, 12, 12)
        assert pixel(interactive, 12, 0) == pixel(focused, 12, 0)
    assert image_for('focus', 'pressed', 'disabled') is image_for('disabled')
