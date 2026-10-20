#!/usr/bin/env python3
"""RIIMA desktop application: state, controls, and the CLI entry point.

This is the only module that imports ``tkinter``. It calls ``riima.core`` for the
automatic analysis, ``riima.review`` for interactive learning and project persistence,
and ``riima.viewer`` for presentation/navigation.
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np

from . import core
from . import viewer
from . import theme
from . import review
from . import metrics
from .exports import export_batch, staged_directory, staged_file
_MASK_MODE_LABELS = {
    "Automática (conservadora)": "auto",
    "Somente sobreposição comum": "overlap_only",
}

# UX safety threshold (not a scientific one): below this fraction of the registered
# geometric overlap, the automatic specimen/annotation heuristic is removing enough area
# that it is worth surfacing to the user, who can switch to "Somente sobreposição comum"
# and compare. Chosen conservatively so ordinary slides never trigger it; not derived from
# petrographic validation.
_MASK_COVERAGE_WARN_RATIO = 0.70

_WORKFLOW_STEPS = [('images', 'Imagens'), ('classify', 'Classificação'), ('organize', 'Seleção e Agrupamento'),
                   ('refine', 'Revisão'), ('results', 'Resultados')]


def gui(_test_hook=None) -> None:
    """Build and run the desktop application.

    ``_test_hook``, when given, is called with this function's local namespace right after
    the widget tree is fully built (instead of entering ``mainloop``), which lets tests
    drive the real widgets directly: same construction, same event bindings, no separate
    test-only code path. Production callers never pass it.
    """
    import sys
    import tkinter as tk
    from tkinter import colorchooser, filedialog, messagebox, simpledialog, ttk
    from PIL import Image, ImageTk

    if sys.platform.startswith("win"):
        try:
            # Windows groups taskbar icons by "AppUserModelID"; without an explicit one,
            # a python.exe/pythonw.exe-launched Tk app inherits Python's own taskbar icon
            # regardless of iconbitmap/iconphoto below. Must run before the first window
            # (tk.Tk()) is created.
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("RIIMA.PetrographicViewer")
        except Exception:
            pass

    root = tk.Tk()
    root.title("RIIMA")
    root.geometry("1280x820")
    root.minsize(1080, 700)
    _app_icon_photo = None
    assets_dir = Path(__file__).resolve().parent / "assets" / "branding"
    try:
        icon_ico = assets_dir / "riima_icon.ico"
        if icon_ico.exists():
            root.iconbitmap(default=str(icon_ico))
    except Exception:
        pass  # .ico title-bar icon is Windows-specific; never block startup on other platforms
    try:
        icon_png = assets_dir / "riima_logo.png"
        if icon_png.exists():
            _app_icon_photo = tk.PhotoImage(file=str(icon_png))
            root.iconphoto(True, _app_icon_photo)
    except Exception:
        pass  # missing/unsupported icon asset must never block the app from opening

    style = ttk.Style(root)
    theme.apply(root, style)

    out_var = tk.StringVar()
    automatic_output_var = tk.BooleanVar(value=True)
    reference_var = tk.IntVar(value=0)
    view_type_var = tk.StringVar(value="layers")  # "layers" (standard) or "compare"
    split_left_var = tk.StringVar()
    split_right_var = tk.StringVar()
    clusters_var = tk.IntVar(value=8)
    mask_mode_var = tk.StringVar(value="Automática (conservadora)")
    status_var = tk.StringVar(value="Selecione uma ou mais imagens para começar.")
    summary_var = tk.StringVar(value="")
    preview_title_var = tk.StringVar(value="Prévia da lâmina")
    review_enabled_var = tk.BooleanVar(value=False)
    review_tool_var = tk.StringVar(value="pan")
    review_image_var = tk.StringVar()
    review_cluster_var = tk.StringVar()
    review_display_var = tk.StringVar(value="Sobreposição")
    workflow_var = tk.StringVar(value='images')
    workflow_hint_var = tk.StringVar()
    isolate_var = tk.BooleanVar(value=False)
    contours_var = tk.BooleanVar(value=False)
    legacy_masks_var = tk.BooleanVar(value=False)
    review_radius_var = tk.IntVar(value=3)
    review_opacity_var = tk.DoubleVar(value=50)
    review_status_var = tk.StringVar()
    show_applied_var = tk.BooleanVar(value=False)
    review_tool_buttons, view_mode_buttons = {}, {}
    def refresh_active_tool_buttons(*_args):
        for tool, button in review_tool_buttons.items():
            button.config(style='Selected.TButton' if tool == review_tool_var.get() else 'TButton')
    review_tool_var.trace_add('write', refresh_active_tool_buttons)
    reference_buttons = {}

    def refresh_reference_buttons(*_args):
        for index, button in reference_buttons.items():
            button.config(style='Selected.TButton' if index == reference_var.get() else 'TButton')
    reference_var.trace_add('write', refresh_reference_buttons)

    session = {
        "images": [],                 # [{"path", "name"}], in picker order
        "_active_vars": [],            # tk.BooleanVar per row, parallel to "images"
        "_status_labels": {},          # name -> ttk.Label, updated after analysis
        "registration_status": {},     # name -> user-facing text
        "prepared": None,
        "photo": None,
        "current_rgb": None,
        "current_image_shape": None,
        "viewport": None,
        "compare_mode": False,
        "split_fraction": 0.5,
        "_compare_sources": {},
        "compare_cache": None,          # {"left_id","right_id","left_rgb","right_rgb"}
        "layer_order": [],
        "layer_states": {},
        "_layer_entries": {},
        "drag_mode": None,
        "drag_last": None,
        "autosave_after": None,
        "resize_after": None,
        "layer_opacity_after": None,
        "analysis_seconds": None,
        "project_path": None,
        "review_after": None,
        "stroke_last": None,
        "workspace_selected_groups": [],
    }
    retained_legend_rows = ()

    def set_status(text: str):
        status_var.set(text)
        root.update_idletasks()

    def selected_workspace_groups():
        prepared = session.get('prepared')
        if prepared is None or 'review' not in prepared:
            return []
        state = prepared['review']
        if workflow_var.get() == 'refine':
            return [state.selected_cluster]
        chosen = [c for c in session['workspace_selected_groups'] if c in state.groups]
        return chosen

    def workspace_settings():
        return dict(step=workflow_var.get(), display=review_display_var.get(),
                    isolate=isolate_var.get(), contours=contours_var.get(),
                    show_applied=show_applied_var.get(), automatic_output=automatic_output_var.get(),
                    selected_groups=list(session['workspace_selected_groups']),
                    opacity=review_opacity_var.get(), view_type=view_type_var.get(),
                    compare_left=split_left_var.get(), compare_right=split_right_var.get(),
                    split_fraction=session['split_fraction'])

    def remember_workspace():
        prepared = session.get('prepared')
        if prepared is not None:
            prepared['workspace'] = workspace_settings()

    def go_to_step(step):
        if step not in dict(_WORKFLOW_STEPS):
            return
        if step in ('organize', 'refine', 'results') and session.get('prepared') is None:
            return
        workflow_var.set(step)
        review_enabled_var.set(step == 'refine')
        review_tool_var.set('pan')
        if step == 'refine':
            isolate_var.set(True)
            review_display_var.set('Sobreposição')
            session['workspace_selected_groups'] = [session['prepared']['review'].selected_cluster]
            refresh_group_listboxes()
        session['compare_cache'] = None
        rebuild_compare_sources()
        change_preview()

    def advance_step():
        keys = [key for key, _ in _WORKFLOW_STEPS]
        index = keys.index(workflow_var.get())
        if index < len(keys) - 1:
            go_to_step(keys[index + 1])

    def previous_step():
        keys = [key for key, _ in _WORKFLOW_STEPS]
        index = keys.index(workflow_var.get())
        if index:
            go_to_step(keys[index - 1])

    def on_group_selection(_event=None):
        prepared = session.get('prepared')
        if prepared is None:
            return
        state = prepared['review']
        raw_members = set().union(*(prepared['interpretation'].groups[i]
                                    for i in optical_group_list.curselection()))
        chosen = [c for c in state.class_ids if state.groups[c] <= raw_members]
        session['workspace_selected_groups'] = chosen
        if chosen:
            state.selected_cluster = chosen[0]
        isolate_var.set(True)
        refresh_review_controls()
        session['compare_cache'] = None
        rebuild_compare_sources()
        change_preview()

    def select_list_item(event):
        """A plain click replaces the selection; Shift toggles one additional item."""
        listbox = event.widget
        if not listbox.size():
            return 'break'
        index = listbox.nearest(event.y)
        selected = index in listbox.curselection()
        if not event.state & 1:
            listbox.selection_clear(0, tk.END)
        if event.state & 1 and selected:
            listbox.selection_clear(index)
        else:
            listbox.selection_set(index)
        listbox.selection_anchor(index)
        listbox.activate(index)
        listbox.focus_set()
        listbox.event_generate('<<ListboxSelect>>', when='tail')
        return 'break'

    def choose_images():
        paths = filedialog.askopenfilenames(
            filetypes=[
                ("Imagens", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"),
                ("Todos os arquivos", "*.*"),
            ]
        )
        if not paths:
            return
        # Finish saving before replacing any input or review state.
        if session.get('prepared') is not None and not save_project_current():
            return
        for timer in ('review_after', 'autosave_after', 'layer_opacity_after'):
            if session.get(timer) is not None:
                root.after_cancel(session[timer])
                session[timer] = None
        session['prepared'] = None
        session['project_path'] = None
        session['compare_mode'] = False
        session['compare_cache'] = None
        session['viewport'] = None
        session['workspace_selected_groups'] = []
        review_image_combo['values'] = []
        review_cluster_combo['values'] = []
        review_image_var.set('')
        review_cluster_var.set('')
        optical_group_list.delete(0, tk.END)
        family_group_list.delete(0, tk.END)
        view_type_var.set('layers')
        summary_var.set('')
        session["images"] = [{"path": p, "name": Path(p).name} for p in paths]
        session["registration_status"] = {}
        reference_var.set(0)
        if automatic_output_var.get() or not out_var.get():
            parent = Path(paths[0]).parent
            directory = parent / 'riima_output'
            suffix = 2
            while directory.exists():
                directory = parent / f'riima_output_{suffix}'
                suffix += 1
            out_var.set(str(directory))
        rebuild_image_rows()
        rebuild_layers(reset=True)
        go_to_step('images')
        status_var.set(
            f"{len(paths)} imagem(ns) selecionada(s) para nova análise. Marque as ativas e a referência, depois clique em Gerar classificação."
        )

    def rebuild_image_rows():
        reference_buttons.clear()
        for child in images_list_frame.winfo_children():
            child.destroy()
        session["_active_vars"] = []
        session["_status_labels"] = {}
        images = session.get("images", [])
        if not images:
            prepared = session.get('prepared')
            if prepared is not None:
                for name in prepared['optical']['image_names']:
                    ttk.Label(images_list_frame, text=name).pack(anchor='w')
                ttk.Label(images_list_frame, text='Imagens incluídas no projeto. Para refazer a análise, selecione os arquivos originais.',
                          wraplength=260, foreground=theme.TEXT_SECONDARY).pack(anchor='w', pady=5)
            else:
                ttk.Label(images_list_frame, text="Nenhuma imagem selecionada.", foreground=theme.TEXT_SECONDARY).pack(anchor="w")
            return
        header = ttk.Frame(images_list_frame)
        header.pack(fill="x")
        ttk.Label(header, text="Usar", width=5, foreground=theme.TEXT_SECONDARY).pack(side="left")
        ttk.Label(header, text="Ref.", width=5, foreground=theme.TEXT_SECONDARY).pack(side="left")
        ttk.Label(header, text="Imagem", foreground=theme.TEXT_SECONDARY).pack(side="left")
        status_map = session.get("registration_status", {})
        for i, meta in enumerate(images):
            row = ttk.Frame(images_list_frame)
            row.pack(fill="x", pady=1)
            active_var = tk.BooleanVar(value=True)
            session["_active_vars"].append(active_var)
            ttk.Checkbutton(row, variable=active_var, width=3).pack(side="left")
            ref_button = ttk.Button(row, text='Ref.', width=4, command=lambda index=i: reference_var.set(index))
            ref_button.pack(side='left', padx=(0, 3))
            reference_buttons[i] = ref_button
            name = meta["name"]
            ttk.Label(row, text=name, width=13, anchor="w").pack(side="left")
            status_lbl = ttk.Label(row, text=status_map.get(name, ""), foreground=theme.TEXT_SECONDARY)
            status_lbl.pack(side="left")
            session["_status_labels"][name] = status_lbl
        refresh_reference_buttons()

    def update_row_statuses(status_map: dict):
        session["registration_status"] = status_map
        for name, label in session.get("_status_labels", {}).items():
            label.config(text=status_map.get(name, ""))

    def choose_out():
        path = filedialog.askdirectory()
        if path:
            automatic_output_var.set(False)
            out_var.set(path)
            if session.get('prepared') is not None:
                session['prepared']['output'] = path
                session['prepared']['output_pending'] = False

    def prepare_output_destination():
        """Claim a fresh resumed destination only when an export is requested."""
        prepared = session.get('prepared')
        if prepared is None or not prepared.get('output_pending', False):
            return True
        target = Path(prepared['output'])
        while True:
            created = False
            try:
                # Exclusive creation also handles another writer claiming the
                # suggestion after load_project returned it.
                target.mkdir(exist_ok=False)
                created = True
                with tempfile.TemporaryFile(dir=target):
                    pass
            except FileExistsError:
                target = Path(review._portable_output(session['project_path'], prepared['output_history']))
                continue
            except OSError as exc:
                if created:
                    try:
                        target.rmdir()  # Remove only our empty, unusable directory.
                    except OSError:
                        pass
                messagebox.showinfo('Pasta de resultados',
                                    f'Não foi possível criar a pasta de resultados: {exc}\nEscolha outra pasta para exportar.')
                selected = filedialog.askdirectory(title='Escolher pasta de resultados')
                if not selected:
                    return False
                target = Path(selected)
                try:
                    with tempfile.TemporaryFile(dir=target):
                        pass
                except OSError as error:
                    messagebox.showerror('Pasta de resultados', str(error))
                    return False
            prepared['output'] = str(target)
            prepared['output_pending'] = False
            out_var.set(str(target))
            return True

    def open_output_folder():
        path = None
        if session.get("prepared") is not None:
            path = Path(session["prepared"]["output"])
        elif out_var.get():
            path = Path(out_var.get())
        if path is None or not path.exists():
            messagebox.showinfo("Pasta de resultados", "A pasta será criada quando a análise for executada.")
            return
        try:
            import os, subprocess
            if sys.platform.startswith("win"):
                os.startfile(str(path))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except Exception as exc:
            messagebox.showerror("Abrir pasta", str(exc))

    def current_interpretation():
        prepared = session.get("prepared")
        return prepared["interpretation"] if prepared is not None else None

    def prepared_for_view() -> dict | None:
        prepared = session.get("prepared")
        if prepared is None or prepared.get("analysis_mode") != "independent":
            return prepared
        state = prepared["review"]
        optical = state.items[state.selected_image]["optical"]
        interpretation = core.Interpretation(optical)
        interpretation._restore(prepared["interpretation"].to_dict())
        return dict(prepared, optical=optical, interpretation=interpretation)

    def setup_review():
        prepared = session.get("prepared")
        if prepared is None:
            return
        state = prepared.setdefault("review", None)
        if state is None:
            state = prepared["review"] = review.ClusterReview.from_prepared(prepared)
        names = ([item['name'] for item in state.items] if state.mode == 'independent'
                 else prepared['optical']['image_names'])
        review_image_combo['values'] = [f'{i + 1}: {name}' for i, name in enumerate(names)]
        review_image_combo.current(state.selected_image)
        saved = prepared.get('workspace', {})
        session['workspace_selected_groups'] = [c for c in saved.get('selected_groups', [state.selected_cluster])
                                                 if c in state.groups]
        refresh_review_controls()

    def review_group_label(group):
        prepared = session.get('prepared')
        state = prepared['review']
        members = state.groups[group]
        source = '+'.join(str(c) for c in sorted(members))
        label = f'Cluster {group}' if len(members) == 1 else f'Grupo {source}'
        interpretation = prepared['interpretation']
        for i, existing in enumerate(interpretation.groups):
            if existing == members:
                name = interpretation.names.get(i)
                return f'{name} ({label})' if name else label
        return label

    def sync_review_groups(*, topology=True):
        prepared = session.get('prepared')
        if prepared is None or 'review' not in prepared:
            return
        state = prepared['review']
        interpretation = prepared['interpretation']
        if topology:
            state.sync_groups(interpretation.groups)
        palette = core.cluster_palette(len(interpretation.groups))
        for i, group in enumerate(interpretation.groups):
            if state.groups.get(min(group)) == group:
                state.palette[min(group)] = interpretation.colors.get(i, palette[i])

    def refresh_review_controls():
        prepared = session.get('prepared')
        if prepared is None or 'review' not in prepared:
            return
        state = prepared['review']
        session['review_group_ids'] = state.class_ids
        review_cluster_combo['values'] = [
            f"{review_group_label(c)} ({'fundo' if c in state.background_groups else 'pronto' if c in state.completed else 'em revisão'})"
            for c in state.class_ids]
        review_cluster_combo.current(state.class_ids.index(state.selected_cluster))
        done = state.selected_cluster in state.completed
        background = state.selected_cluster in state.background_groups
        review_approve_btn.config(state='disabled' if done or background or state.has_pending else 'normal')
        review_reopen_btn.config(state='normal' if done else 'disabled')
        review_update_btn.config(state='normal' if state.has_pending else 'disabled')
        review_undo_btn.config(state='normal' if state.can_undo else 'disabled')
        review_reset_btn.config(state='disabled' if done else 'normal')
        for tool, button in review_tool_buttons.items():
            button.config(state='disabled' if (done or background) and tool in ('positive', 'negative', 'erase') else 'normal',
                          style='Selected.TButton' if tool == review_tool_var.get() else 'TButton')
        counts = [int((item['annotations'].get(state.selected_cluster, np.zeros(0)) == value).sum())
                  for value in (1, -1) for item in state.items]
        n = len(state.items)
        review_status_var.set(
            f'{len(state.completed)} grupos prontos, {len(state.background_groups)} de fundo. Seleção: {state.counts(state.selected_cluster)} pixels.\n'
            f'Marcações: {sum(counts[:n])} manter, {sum(counts[n:])} remover.'
            + ('\nEdições pendentes: clique em Atualizar classificação.' if state.has_pending else ''))

    def clear_group_selection():
        optical_group_list.selection_clear(0, tk.END)
        family_group_list.selection_clear(0, tk.END)
        on_group_selection()

    def undo_review_stroke():
        prepared = session.get('prepared')
        if prepared is None or not prepared['review'].undo():
            return
        refresh_review_controls()
        refresh_group_listboxes()
        refresh_preview_content()
        status_var.set('Última pincelada desfeita. Atualize a classificação se houver edições pendentes.')

    def selected_groups_background(value):
        prepared = session.get('prepared')
        chosen = selected_workspace_groups()
        if prepared is None or not chosen:
            status_var.set('Selecione ao menos um grupo para alterar o fundo.')
            return
        try:
            prepared['review'].set_background(chosen, value)
        except ValueError as exc:
            messagebox.showinfo('Fundo da análise', str(exc))
            return
        refresh_group_listboxes()
        refresh_review_controls()
        refresh_preview_content()
        status_var.set('Seleção enviada ao fundo. Clique em Atualizar classificação na Revisão.' if value else
                       'Seleção restaurada. Clique em Atualizar classificação na Revisão.')

    def reset_review_group():
        prepared = session.get('prepared')
        if prepared is None:
            return
        try:
            prepared['review'].reset_group(prepared['review'].selected_cluster)
        except ValueError as exc:
            messagebox.showinfo('Resetar grupo', str(exc))
            return
        refresh_group_listboxes()
        refresh_review_controls()
        refresh_preview_content()
        status_var.set('Marcações do grupo limpas. Clique em Atualizar classificação para reiniciar seu aprendizado.')

    def select_review_image(_event=None):
        prepared = session.get('prepared')
        if prepared is None or review_image_combo.current() < 0:
            return
        prepared['review'].selected_image = review_image_combo.current()
        session['compare_cache'] = None
        rebuild_layers()
        rebuild_compare_sources()
        refresh_preview_content()

    def select_review_cluster(_event=None):
        prepared = session.get('prepared')
        if prepared is None or review_cluster_combo.current() < 0:
            return
        cluster = session['review_group_ids'][review_cluster_combo.current()]
        if session.get('review_after') is not None:
            root.after_cancel(session['review_after'])
            session['review_after'] = None
        prepared['review'].selected_cluster = cluster
        session['workspace_selected_groups'] = [cluster]
        isolate_var.set(True)
        refresh_group_listboxes()
        refresh_review_controls()
        session['compare_cache'] = None
        rebuild_compare_sources()
        refresh_preview_content()

    def refine_review_cluster():
        if session.get('review_after') is not None:
            root.after_cancel(session['review_after'])
        session['review_after'] = None
        prepared = session.get('prepared')
        if prepared is None or 'review' not in prepared:
            return
        state = prepared['review']
        review_update_btn.config(state='disabled')
        set_status('Aplicando as edições pendentes ao conjunto de imagens...')
        try:
            trained = state.refine_pending(preferred=state.selected_cluster)
            show_applied_var.set(False)
            refresh_review_controls()
            refresh_group_listboxes()
            session['compare_cache'] = None
            rebuild_layers()
            rebuild_compare_sources()
            refresh_preview_content()
            status_var.set('Classificação atualizada com todas as edições pendentes.' if any(trained.values()) else
                           'Marcações aplicadas. Manter e Remover refinam os grupos no conjunto.')
        except Exception as exc:
            messagebox.showerror('Revisão do grupo', str(exc))
        finally:
            refresh_review_controls()

    def approve_review_cluster():
        prepared = session.get('prepared')
        if prepared is None:
            return
        if session.get('review_after') is not None:
            root.after_cancel(session['review_after'])
            session['review_after'] = None
        state = prepared['review']
        try:
            state.approve(state.selected_cluster)
        except Exception as exc:
            messagebox.showerror('Grupo pronto', str(exc))
            return
        refresh_review_controls()
        refresh_group_listboxes()
        session['compare_cache'] = None
        rebuild_layers()
        rebuild_compare_sources()
        refresh_preview_content()
        status_var.set('Grupo aprovado. Os pixels da seleção ficam protegidos.')

    def reopen_review_cluster():
        prepared = session.get('prepared')
        if prepared is None:
            return
        state = prepared['review']
        try:
            state.reopen(state.selected_cluster)
        except ValueError as exc:
            messagebox.showinfo('Reabrir grupo', str(exc))
            return
        refresh_review_controls()
        refresh_group_listboxes()
        refresh_preview_content()
        status_var.set('Grupo reaberto para revisão.')

    def save_project_current(*, choose=False):
        prepared = session.get('prepared')
        if prepared is None:
            return False
        remember_workspace()
        path = session.get('project_path')
        if choose:
            path = filedialog.asksaveasfilename(defaultextension='.riima', initialfile='projeto.riima',
                                              filetypes=[('Projeto RIIMA', '*.riima')])
            if not path:
                return False
        path = path or str(Path(prepared['output']) / 'projeto.riima')
        try:
            review.save_project(prepared, path)
            session['project_path'] = path
            status_var.set(f'Projeto salvo: {path}')
            return True
        except Exception as exc:
            messagebox.showerror('Salvar projeto', str(exc))
            return False

    def open_project():
        nonlocal session, retained_legend_rows
        path = filedialog.askopenfilename(filetypes=[('Projeto RIIMA', '*.riima'), ('Todos os arquivos', '*.*')])
        if not path:
            return
        try:
            prepared = review.load_project(path)
            if session.get('prepared') is not None:
                old = session['prepared']
                old_path = session.get('project_path') or str(Path(old['output']) / 'projeto.riima')
                review.save_project(dict(old, workspace=workspace_settings()), old_path)
                if Path(path).resolve() == Path(old_path).resolve():
                    prepared = review.load_project(path)
        except Exception as exc:
            messagebox.showerror('Abrir projeto', str(exc))
            return
        previous = session
        controls = [(variable, variable.get()) for variable in project_view_variables]
        active = [variable.get() for variable in previous['_active_vars']]
        choices = [(combo, tuple(combo['values'])) for combo in
                   (review_image_combo, review_cluster_combo, split_left_combo, split_right_combo)]
        selections = [(box, box.get(0, tk.END), box.curselection())
                      for box in (optical_group_list, family_group_list)]
        buttons = [(button, {key: button.cget(key) for key in ('state', 'style', 'text')})
                   for button in project_view_buttons]
        legend = [(child, child.pack_info()) for child in legend_rows.winfo_children()]
        canvas_items = [(preview_canvas.type(item), preview_canvas.coords(item),
                         {key: option[-1] for key, option in preview_canvas.itemconfigure(item).items()})
                        for item in preview_canvas.find_all()]
        candidate = dict(previous)
        retained_legend_rows = tuple(child for child, _ in legend)
        try:
            for child, _ in legend:
                child.pack_forget()
            session = candidate
            show_loaded_project(prepared, path)
        except Exception as exc:
            session = previous
            for variable, value in controls:
                variable.set(value)
            rebuild_image_rows()
            for variable, value in zip(session['_active_vars'], active):
                variable.set(value)
            refresh_review_controls()
            for combo, values in choices:
                combo['values'] = values
            for box, entries, selected in selections:
                box.delete(0, tk.END)
                for entry in entries:
                    box.insert(tk.END, entry)
                for index in selected:
                    box.selection_set(index)
            update_context_controls()
            clear_legend()
            for child, layout in legend:
                child.pack(**layout)
            retained_legend_rows = ()
            for button, options in buttons:
                button.configure(**options)
            for variable, value in controls:
                variable.set(value)
            preview_canvas.delete('all')
            for kind, coordinates, options in canvas_items:
                getattr(preview_canvas, f'create_{kind}')(*coordinates, **options)
            messagebox.showerror('Abrir projeto', str(exc))
            return
        for child, _ in legend:
            child.destroy()
        retained_legend_rows = ()
        for timer in ('review_after', 'autosave_after'):
            if previous.get(timer) is not None:
                root.after_cancel(previous[timer])
                candidate[timer] = None
        previous.clear()
        previous.update(candidate)
        session = previous

    def show_loaded_project(prepared, path):
        session['prepared'] = prepared
        session['project_path'] = path
        session['compare_cache'] = None
        session['viewport'] = None
        out_var.set(prepared['output'])
        clusters_var.set(prepared['config'].clusters)
        session['images'] = ([dict(path=item['path'], name=item['name']) for item in prepared['review'].items]
                             if prepared['review'].mode == 'independent' else [])
        if prepared['review'].mode == 'registered':
            entries = prepared['optical'].get('registered_images', [])
            if entries and all(entry.get('path') for entry in entries):
                session['images'] = [dict(path=entry['path'], name=entry['name']) for entry in entries]
        reference_var.set(prepared['optical']['reference_index'])
        mask_mode_var.set(next(label for label, mode in _MASK_MODE_LABELS.items()
                              if mode == prepared['optical']['mask_mode']))
        rebuild_image_rows()
        review_tool_var.set('pan')
        setup_review()
        refresh_group_listboxes()
        saved_workspace = prepared.get('workspace', {})
        automatic_output_var.set(bool(saved_workspace.get('automatic_output', True)))
        workflow_var.set(saved_workspace.get('step', 'refine' if prepared['review'].completed or prepared['review'].has_pending else 'organize'))
        if workflow_var.get() not in dict(_WORKFLOW_STEPS):
            workflow_var.set('organize')
        review_enabled_var.set(workflow_var.get() == 'refine')
        review_display_var.set(saved_workspace.get('display', 'Sobreposição'))
        isolate_var.set(bool(saved_workspace.get('isolate', True)))
        contours_var.set(bool(saved_workspace.get('contours', False)))
        show_applied_var.set(bool(saved_workspace.get('show_applied', False)))
        review_opacity_var.set(float(saved_workspace.get('opacity', 50)))
        view_type_var.set('compare' if saved_workspace.get('view_type') == 'compare' else 'layers')
        session['compare_mode'] = view_type_var.get() == 'compare'
        split_left_var.set(saved_workspace.get('compare_left', 'Imagem original'))
        split_right_var.set(saved_workspace.get('compare_right', 'Classificação atual'))
        session['split_fraction'] = float(np.clip(saved_workspace.get('split_fraction', .5), 0, 1))
        rebuild_layers(reset=True)
        rebuild_compare_sources()
        update_summary_text()
        if session['compare_mode']:
            compare_panel.pack(fill='x')
        else:
            compare_panel.pack_forget()
        change_preview()
        status_var.set('Projeto retomado com marcações, máscaras aprovadas e aprendizado preservados.')

    def close_app():
        if session.get('prepared') is not None and not save_project_current():
            return
        root.destroy()

    def refresh_group_listboxes():
        optical_group_list.delete(0, "end")
        family_group_list.delete(0, "end")
        interpretation = current_interpretation()
        if interpretation is None:
            return
        state = session['prepared']['review']
        session['workspace_selected_groups'] = [c for c in session['workspace_selected_groups'] if c in state.groups]
        for i, group in enumerate(interpretation.groups):
            src = "+".join(str(v) for v in sorted(group))
            base = f"Cluster {src}" if len(group) == 1 else f"Grupo {i} [clusters {src}]"
            name = interpretation.names.get(i)
            state = session['prepared']['review']
            ids = [c for c in state.class_ids if state.groups[c] <= group]
            status = ('fundo' if ids and all(c in state.background_groups for c in ids) else
                      'pronto' if ids and all(c in state.completed for c in ids) else
                      'edições pendentes' if any(c in state.dirty for c in ids) else 'em revisão')
            optical_group_list.insert("end", f"{name or f'Grupo {min(group)}'} · {status}")
            if any(c in session['workspace_selected_groups'] for c in ids):
                optical_group_list.selection_set(i)
        for i, family in enumerate(interpretation.families):
            src = "+".join(str(v) for v in sorted(family["clusters"]))
            name = family.get("name") or f"Família {i + 1}"
            family_group_list.insert("end", f"{name} [clusters {src}]")

    def clear_legend():
        for child in legend_rows.winfo_children():
            if child not in retained_legend_rows:
                child.destroy()

    def build_preview_entries(prepared):
        return viewer.workspace_sources(prepared, selected_workspace_groups())

    def rebuild_compare_sources():
        prepared = prepared_for_view()
        if prepared is None:
            session["_compare_sources"] = {}
            split_left_combo["values"] = []
            split_right_combo["values"] = []
            return
        entries = build_preview_entries(prepared)
        session["_compare_sources"] = {e["label"]: e for e in entries}
        labels = [e["label"] for e in entries]
        split_left_combo["values"] = labels
        split_right_combo["values"] = labels
        if split_left_var.get() not in labels:
            split_left_var.set(labels[0])
        if split_right_var.get() not in labels:
            split_right_var.set(labels[1] if len(labels) > 1 else labels[0])

    def refresh_compare_cache():
        """(Re-)render the two compare sources only when the *selection* changed.

        This is the main fix for slow divider dragging: dragging never calls a source's
        ``render()`` again; it only recombines these two already-rendered arrays.
        """
        sources = session.get("_compare_sources") or {}
        left_entry = sources.get(split_left_var.get())
        right_entry = sources.get(split_right_var.get())
        if left_entry is None or right_entry is None:
            rebuild_compare_sources()
            sources = session.get("_compare_sources") or {}
            left_entry = sources.get(split_left_var.get())
            right_entry = sources.get(split_right_var.get())
        if left_entry is None or right_entry is None:
            session["compare_cache"] = None
            return None, None
        cache = session.get("compare_cache")
        if cache and cache["left_id"] == left_entry["id"] and cache["right_id"] == right_entry["id"]:
            return left_entry, right_entry
        session["compare_cache"] = {
            "left_id": left_entry["id"], "right_id": right_entry["id"],
            "left_rgb": left_entry["render"](), "right_rgb": right_entry["render"](),
        }
        return left_entry, right_entry



    def rebuild_layers(*, reset: bool = False):
        # A fixed workspace replaces the visible stack of generated products.
        for child in layers_list_frame.winfo_children():
            child.destroy()
        prepared = prepared_for_view()
        entries = build_preview_entries(prepared) if prepared is not None else []
        session['_layer_entries'] = {entry['id']: entry for entry in entries}
        session['layer_order'] = [entry['id'] for entry in entries]
        session['layer_states'] = {entry['id']: dict(visible=entry['id'] == 'classification', opacity=1.)
                                   for entry in entries}

    def _visible_layer_ids():
        order = session.get("layer_order", [])
        states = session.get("layer_states", {})
        return [
            lid for lid in order
            if states.get(lid, {}).get("visible") and float(states.get(lid, {}).get("opacity", 0.0)) > 0.0
        ]

    def _draw_legend_rows(mode, rows):
        for i, row in enumerate(rows):
            line = ttk.Frame(legend_rows)
            line.pack(fill="x", pady=theme.SPACE_XS)
            color = row.get("color")
            if color is not None:
                swatch = tk.Label(
                    line, width=2, height=1,
                    bg=f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}",
                    relief="solid", bd=1,
                )
                swatch.pack(side="left", padx=(0, 7), anchor="n")
            text = ttk.Frame(line)
            text.pack(side="left", fill="x", expand=True)
            name = ttk.Label(text, text=row["label"], font=theme.font('body', bold=True),
                             wraplength=190, justify='left', anchor='w')
            name.pack(anchor="w", fill='x')
            detail = ttk.Label(text, text=row.get("detail", ""), wraplength=190,
                               foreground=theme.TEXT_SECONDARY, justify='left', anchor='w')
            detail.pack(anchor="w", fill='x')
            # Wrap by the allocated text column, keeping complete names in the
            # vertical scroller even with a wide fallback font or long group name.
            text.bind('<Configure>', lambda event, labels=(name, detail):
                      [label.configure(wraplength=max(1, event.width)) for label in labels])



    def draw_legend():
        clear_legend()
        prepared = prepared_for_view()
        if prepared is None:
            ttk.Label(legend_rows, text='Importe imagens e gere a classificação para começar.',
                      wraplength=210, foreground=theme.TEXT_SECONDARY).pack(anchor='w')
            return
        chosen = selected_workspace_groups()
        if session.get('compare_mode'):
            for side, label in [('Esquerda', split_left_var.get()), ('Direita', split_right_var.get())]:
                ttk.Label(legend_rows, text=f'{side}: {label}', font=theme.font('body', bold=True),
                          wraplength=210).pack(anchor='w', pady=(8, 2))
                if label != 'Imagem original':
                    _draw_legend_rows(label, viewer.workspace_legend(prepared,
                                      chosen if label == 'Máscara selecionada' else None,
                                      mask=label == 'Máscara selecionada'))
                else:
                    ttk.Label(legend_rows, text=review_image_var.get(), wraplength=210).pack(anchor='w')
        else:
            ttk.Label(legend_rows, text=review_image_var.get(), font=theme.font('body', bold=True),
                      wraplength=210).pack(anchor='w', pady=(0, 6))
            display = review_display_var.get()
            selected = chosen if display == 'Máscara selecionada' or (
                isolate_var.get() and display in ('Sobreposição', 'Grupo selecionado')) else None
            if selected == []:
                ttk.Label(legend_rows, text='Selecione um ou mais grupos em Seleção e Agrupamento.', wraplength=210).pack(anchor='w')
            _draw_legend_rows('current', viewer.workspace_legend(prepared, selected,
                                                               mask=display == 'Máscara selecionada'))
        if review_enabled_var.get():
            ttk.Label(legend_rows, text='Verde: manter. Vermelho: remover.\nRoxo: fundo. Azul: restaurar/apagar.\nGrupos prontos estão protegidos.',
                      wraplength=210).pack(anchor='w', pady=8)

    def draw_divider_handle(fraction, image_shape, viewport_size):
        """Comparator divider: a line spanning only the displayed image height, with a
        plain circular handle (a wide invisible hitbox handles grabbing it, see
        split_divider_geometry) that does not dominate the image underneath it."""
        geo = viewer.split_divider_geometry(session["viewport"], image_shape, viewport_size, fraction)
        x = geo["canvas_x"]
        preview_canvas.create_line(x, geo["y_top"], x, geo["y_bottom"], fill=theme.BG_PANEL, width=2)
        hx, hy = geo["handle_center"]
        r = 10
        preview_canvas.create_oval(hx - r, hy - r, hx + r, hy + r, fill=theme.BG_PANEL, outline=theme.ACCENT, width=2)

    def render_preview_canvas():
        preview_canvas.delete("all")
        rgb = session.get("current_rgb")
        width = max(1, int(preview_canvas.winfo_width()))
        height = max(1, int(preview_canvas.winfo_height()))
        if rgb is None:
            preview_canvas.create_text(
                width // 2, height // 2,
                text="Selecione imagens e clique em Analisar.",
                fill=theme.TEXT_ON_CANVAS, font=theme.font('empty'),
            )
            return
        image_shape = rgb.shape[:2]
        viewport_size = (width, height)
        if session.get("viewport") is None:
            session["viewport"] = viewer.ViewportState.fit(image_shape, viewport_size)
        state = session["viewport"]
        plan = viewer.viewport_render_plan(state, image_shape, viewport_size)
        x0, y0, x1, y1 = plan["src_box"]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(image_shape[1], x1), min(image_shape[0], y1)
        if x1 <= x0 or y1 <= y0:
            return
        crop = rgb[y0:y1, x0:x1]
        dst_w, dst_h = max(1, plan["dst_size"][0]), max(1, plan["dst_size"][1])
        pil_img = Image.fromarray(crop)
        if (dst_w, dst_h) != (crop.shape[1], crop.shape[0]):
            resample = Image.Resampling.NEAREST if state.zoom > 1.0 else Image.Resampling.BILINEAR
            pil_img = pil_img.resize((dst_w, dst_h), resample)
        session["photo"] = ImageTk.PhotoImage(pil_img)
        ox, oy = plan["canvas_offset"]
        preview_canvas.create_image(ox, oy, image=session["photo"], anchor="nw")
        if session.get("compare_mode"):
            draw_divider_handle(session.get("split_fraction", 0.5), image_shape, viewport_size)

    def draw_compare_frame(fraction):
        """Fast path used while actively dragging the divider: crops the two *cached*
        full-resolution sources to the current viewport before compositing, so per-frame
        cost is proportional to the visible viewport, not the full image resolution.
        ``session["current_rgb"]`` is deliberately left untouched here; a normal
        ``refresh_preview_content()`` call on release brings it back in sync for
        legend/export purposes.
        """
        cache = session.get("compare_cache")
        shape = session.get("current_image_shape")
        if not cache or shape is None or session.get("viewport") is None:
            return
        width = max(1, int(preview_canvas.winfo_width()))
        height = max(1, int(preview_canvas.winfo_height()))
        composed, plan = viewer.render_compare_view(
            cache["left_rgb"], cache["right_rgb"], fraction, session["viewport"], shape, (width, height)
        )
        dst_w, dst_h = max(1, plan["dst_size"][0]), max(1, plan["dst_size"][1])
        pil_img = Image.fromarray(composed)
        if (dst_w, dst_h) != (composed.shape[1], composed.shape[0]):
            resample = Image.Resampling.NEAREST if session["viewport"].zoom > 1.0 else Image.Resampling.BILINEAR
            pil_img = pil_img.resize((dst_w, dst_h), resample)
        preview_canvas.delete("all")
        session["photo"] = ImageTk.PhotoImage(pil_img)
        ox, oy = plan["canvas_offset"]
        preview_canvas.create_image(ox, oy, image=session["photo"], anchor="nw")
        draw_divider_handle(fraction, shape, (width, height))



    def refresh_preview_content(*, legend: bool = True):
        prepared = prepared_for_view()
        if prepared is None:
            session['current_rgb'] = None
            session['current_image_shape'] = None
            preview_title_var.set('Imagens do projeto')
            render_preview_canvas()
            if legend:
                draw_legend()
            return
        remember_workspace()
        if session.get('compare_mode'):
            refresh_compare_cache()
            cache = session.get('compare_cache')
            rgb = viewer.compose_split(cache['left_rgb'], cache['right_rgb'], session.get('split_fraction', .5))
            preview_title_var.set(f'Comparação: {split_left_var.get()} | {split_right_var.get()}')
        else:
            display = review_display_var.get()
            selected = selected_workspace_groups() if isolate_var.get() or display == 'Máscara selecionada' else None
            annotation_group = prepared['review'].selected_cluster if review_enabled_var.get() else None
            rgb = viewer.render_workspace(prepared, display, selected_groups=selected,
                                          opacity=review_opacity_var.get() / 100,
                                          contours=contours_var.get(), annotation_group=annotation_group,
                                          show_applied=show_applied_var.get())
            preview_title_var.set(f'{review_image_var.get()} | {display}')
        session['current_rgb'] = rgb
        new_shape = rgb.shape[:2]
        if session.get('current_image_shape') != new_shape:
            session['viewport'] = None
        session['current_image_shape'] = new_shape
        render_preview_canvas()
        if legend:
            draw_legend()

    def schedule_redraw(_event=None):
        if session.get("resize_after") is not None:
            root.after_cancel(session["resize_after"])
        session["resize_after"] = root.after(100, render_preview_canvas)

    def _current_divider_geometry():
        state = session.get("viewport")
        shape = session.get("current_image_shape")
        if not session.get("compare_mode") or state is None or shape is None:
            return None
        width = max(1, int(preview_canvas.winfo_width()))
        height = max(1, int(preview_canvas.winfo_height()))
        return viewer.split_divider_geometry(state, shape, (width, height), session.get("split_fraction", 0.5))

    def on_canvas_press(event):
        prepared = session.get('prepared')
        if (review_enabled_var.get() and review_tool_var.get() != 'pan'
                and prepared is not None and 'review' in prepared and not session.get('compare_mode')
                and session.get('viewport') is not None):
            state = prepared['review']
            if ((state.selected_cluster in state.completed or state.selected_cluster in state.background_groups)
                    and review_tool_var.get() not in ('background', 'restore')):
                status_var.set('Reabra o grupo pronto antes de marcar.')
                return
            if contours_var.get():
                contours_var.set(False)
                refresh_preview_content(legend=False)
            session['drag_mode'] = 'annotate'
            session['stroke_last'] = None
            state.begin_stroke()
            paint_review_stroke(event)
            return
        geo = _current_divider_geometry()
        hit = geo is not None and viewer.point_hits_divider(event.x, event.y, geo)
        session["drag_mode"] = "divider" if hit else "pan"
        session["drag_last"] = (event.x, event.y)

    def on_canvas_drag(event):
        shape = session.get("current_image_shape")
        if session.get("drag_mode") is None or shape is None:
            return
        width = max(1, int(preview_canvas.winfo_width()))
        height = max(1, int(preview_canvas.winfo_height()))
        if session['drag_mode'] == 'annotate':
            paint_review_stroke(event)
        elif session["drag_mode"] == "divider":
            img_x, _ = viewer.canvas_to_image((event.x, 0), session["viewport"], shape, (width, height))
            session["split_fraction"] = min(1.0, max(0.0, img_x / max(1, shape[1])))
            draw_compare_frame(session["split_fraction"])
        else:
            last_x, last_y = session["drag_last"]
            dx, dy = event.x - last_x, event.y - last_y
            session["drag_last"] = (event.x, event.y)
            session["viewport"] = viewer.pan_view(session["viewport"], dx, dy, shape, (width, height))
            render_preview_canvas()

    def on_canvas_release(_event=None):
        if session.get('drag_mode') == 'annotate':
            session['prepared']['review'].end_stroke()
            session['stroke_last'] = None
            refresh_review_controls()
            refresh_group_listboxes()
            status_var.set('Marcações acumuladas. Clique em Atualizar classificação quando terminar.')
        was_divider = session.get("drag_mode") == "divider"
        session["drag_mode"] = None
        session["drag_last"] = None
        if was_divider:
            # Synchronize the full composite for the final legend, title and export.
            refresh_preview_content(legend=True)

    def paint_review_stroke(event):
        prepared = session['prepared']
        state = prepared['review']
        shape = session['current_image_shape']
        viewport_size = (max(1, preview_canvas.winfo_width()), max(1, preview_canvas.winfo_height()))
        point = viewer.canvas_to_image((event.x, event.y), session['viewport'], shape, viewport_size)
        indices = viewer.brush_indices(shape, session.get('stroke_last') or point, point, review_radius_var.get())
        session['stroke_last'] = point
        image_id = state.selected_image if state.mode == 'independent' else 0
        tool = review_tool_var.get()
        try:
            if tool in ('background', 'restore'):
                state.exclude(image_id, indices, tool == 'background')
            else:
                state.annotate(image_id, state.selected_cluster, indices,
                               {'positive': 1, 'negative': -1, 'erase': 0}[tool])
        except ValueError as exc:
            status_var.set(str(exc))
            return
        # Change only the affected RGB pixels; do not render/count the dataset per motion.
        item = state.items[image_id]
        allowed = item['valid'].ravel()[indices] & (item['locked'].ravel()[indices] < 0)
        indices = indices[allowed]
        display = review_display_var.get()
        selected = selected_workspace_groups() if isolate_var.get() or display == 'Máscara selecionada' else None
        rgb = viewer.workspace_pixels(prepared, display, indices=indices, selected_groups=selected,
                                      opacity=review_opacity_var.get() / 100,
                                      annotation_group=state.selected_cluster, show_applied=show_applied_var.get())
        session['current_rgb'].reshape(-1, 3)[indices] = rgb
        render_preview_canvas()

    def on_canvas_motion(event):
        """Cursor feedback only: a double-headed arrow near the divider, a hand while panning."""
        geo = _current_divider_geometry()
        if review_enabled_var.get() and review_tool_var.get() != 'pan' and not session.get('compare_mode'):
            preview_canvas.config(cursor='crosshair')
        elif geo is not None and viewer.point_hits_divider(event.x, event.y, geo):
            preview_canvas.config(cursor="sb_h_double_arrow")
        else:
            preview_canvas.config(cursor="fleur" if session.get("current_rgb") is not None else "arrow")

    def on_canvas_wheel(event):
        shape = session.get("current_image_shape")
        if shape is None or session.get("viewport") is None:
            return
        width = max(1, int(preview_canvas.winfo_width()))
        height = max(1, int(preview_canvas.winfo_height()))
        delta = getattr(event, "delta", 0)
        if delta == 0:
            delta = 120 if getattr(event, "num", 5) == 4 else -120
        factor = 1.1 if delta > 0 else 1.0 / 1.1
        session["viewport"] = viewer.zoom_about_point(session["viewport"], factor, (event.x, event.y), shape, (width, height))
        render_preview_canvas()

    def on_canvas_double_click(_event=None):
        set_fit_mode()

    def zoom_button(factor):
        shape = session.get("current_image_shape")
        if shape is None or session.get("viewport") is None:
            return
        width = max(1, int(preview_canvas.winfo_width()))
        height = max(1, int(preview_canvas.winfo_height()))
        cursor = (width / 2.0, height / 2.0)
        session["viewport"] = viewer.zoom_about_point(session["viewport"], factor, cursor, shape, (width, height))
        render_preview_canvas()

    def set_fit_mode():
        shape = session.get("current_image_shape")
        if shape is None:
            return
        width = max(1, int(preview_canvas.winfo_width()))
        height = max(1, int(preview_canvas.winfo_height()))
        session["viewport"] = viewer.ViewportState.fit(shape, (width, height))
        render_preview_canvas()

    def set_width_mode():
        shape = session.get("current_image_shape")
        if shape is None:
            return
        width = max(1, int(preview_canvas.winfo_width()))
        height = max(1, int(preview_canvas.winfo_height()))
        session["viewport"] = viewer.ViewportState.width(shape, (width, height))
        render_preview_canvas()

    def set_100_mode():
        shape = session.get("current_image_shape")
        if shape is None:
            return
        session["viewport"] = viewer.ViewportState.for_100_percent(shape)
        render_preview_canvas()



    def on_view_type_change():
        session['compare_mode'] = view_type_var.get() == 'compare'
        layers_panel.pack_forget()
        if session['compare_mode']:
            display_controls.pack_forget()
            rebuild_compare_sources()
            session['compare_cache'] = None
            compare_panel.pack(fill='x')
        else:
            compare_panel.pack_forget()
            display_controls.pack(fill='x', pady=4)
        change_preview()

    def on_compare_source_change(_event=None):
        session["compare_cache"] = None
        change_preview()

    def schedule_workspace_redraw(_value=None):
        if session.get('layer_opacity_after') is not None:
            root.after_cancel(session['layer_opacity_after'])
        def redraw():
            session['layer_opacity_after'] = None
            change_preview()
        session['layer_opacity_after'] = root.after(80, redraw)



    def update_context_controls():
        prepared = session.get('prepared')
        step = workflow_var.get()
        hints = dict(images='Importe as imagens e escolha como elas serão analisadas.',
                     classify='Gere os grupos automáticos. Depois organize o resultado.',
                     organize='Selecione grupos para inspecionar suas máscaras, unir, nomear e escolher cores.',
                     refine='Marque o que manter ou remover. Atualize ao terminar e aprove cada grupo.',
                     results='Confira o progresso e exporte a classificação atual ou as máscaras selecionadas.')
        workflow_hint_var.set(hints[step])
        for key, frame in step_frames.items():
            if key == step:
                frame.pack(fill='x', pady=(8, 0))
            else:
                frame.pack_forget()
        family_group_frame.pack_forget()
        if prepared is not None and prepared['interpretation'].families:
            legacy_masks_toggle.pack(fill='x', pady=(8, 0))
            if step == 'organize' and legacy_masks_var.get():
                family_group_frame.pack(fill='x', pady=(8, 0))
        else:
            legacy_masks_toggle.pack_forget()
        layers_panel.pack_forget()
        display_controls.pack_forget()
        compare_panel.pack_forget()
        if session.get('compare_mode'):
            compare_panel.pack(fill='x')
        else:
            display_controls.pack(fill='x', pady=4)
        for mode, button in view_mode_buttons.items():
            button.config(style='Selected.TButton' if mode == view_type_var.get() else 'TButton')
        if prepared is not None:
            diagnostics_frame.pack(fill='x', pady=(8, 0))
        else:
            diagnostics_frame.pack_forget()
        for key, button in step_buttons.items():
            available = key == 'images' or (key == 'classify' and bool(session['images'] or prepared)) or prepared is not None
            button.config(state='normal' if available else 'disabled',
                          style='Primary.TButton' if key == step else 'TButton')
        index = [key for key, _ in _WORKFLOW_STEPS].index(step)
        back_btn.config(state='disabled' if index == 0 else 'normal')
        next_available = index < 4 and (bool(session['images']) if step == 'images' else prepared is not None)
        next_btn.config(state='normal' if next_available else 'disabled', text='Próximo')
        if prepared is not None:
            state = prepared['review']
            results_status_var.set(f'{len(state.completed)} grupos prontos e {len(state.background_groups)} de fundo.\n'
                                   + ('Há edições pendentes de atualização.' if state.has_pending else 'Classificação atualizada.'))

    def change_preview():
        update_context_controls()
        refresh_preview_content(legend=True)

    def update_summary_text():
        if session.get("prepared") is not None:
            summary_var.set(viewer.human_analysis_summary(session["prepared"]))

    def autosave_current():
        session["autosave_after"] = None
        if session.get("prepared") is None:
            return
        if not prepare_output_destination():
            return False
        prepared = session['prepared']
        previous_path = session.get('project_path')
        missing = object()
        previous_workspace = prepared.get('workspace', missing)
        try:
            with export_batch():
                staged_directory(prepared['output'])
                if 'review' in prepared and not save_project_current():
                    if previous_workspace is missing:
                        prepared.pop('workspace', None)
                    else:
                        prepared['workspace'] = previous_workspace
                    return False
                if prepared.get('analysis_mode') == 'independent':
                    for i, item in enumerate(prepared['review'].items):
                        interpretation = core.Interpretation(item['optical'])
                        interpretation._restore(prepared['interpretation'].to_dict())
                        core.save_session_outputs(dict(prepared, optical=item['optical'], interpretation=interpretation,
                                                       output=str(Path(prepared['output']) / 'initial' / f'image_{i:03d}')))
                else:
                    core.save_session_outputs(prepared)
                if 'review' in prepared:
                    review.save_review_outputs(prepared)
            status_var.set("Resultados atualizados automaticamente.")
            return True
        except Exception as exc:
            session['project_path'] = previous_path
            if previous_workspace is missing:
                prepared.pop('workspace', None)
            else:
                prepared['workspace'] = previous_workspace
            status_var.set("A análise está pronta, mas o salvamento automático falhou.")
            messagebox.showerror("Salvar resultados", str(exc))
            return False

    def schedule_autosave():
        if session.get("autosave_after") is not None:
            root.after_cancel(session["autosave_after"])
        session["autosave_after"] = root.after(700, save_interpretation_metadata)

    def save_interpretation_metadata():
        session['autosave_after'] = None
        prepared = session.get('prepared')
        if prepared is None:
            return
        if not prepare_output_destination():
            return False
        try:
            folder = Path(prepared['output'])
            with export_batch():
                staged_file(folder / 'interpretation.json').write_text(
                    json.dumps(prepared['interpretation'].to_dict(), indent=2, ensure_ascii=False), encoding='utf-8')
        except OSError as exc:
            status_var.set(f'Salvamento de interpretação pendente: {exc}')

    def refresh_interpretation():
        refresh_review_controls()
        refresh_group_listboxes()
        update_summary_text()
        session["compare_cache"] = None
        rebuild_layers()
        rebuild_compare_sources()
        change_preview()
        schedule_autosave()

    def apply_interpretation(method, *args):
        interpretation = current_interpretation()
        if interpretation is None:
            return
        previous = interpretation.to_dict()
        history = list(interpretation._history)
        try:
            getattr(interpretation, method)(*args)
            sync_review_groups(topology=previous['groups'] != interpretation.to_dict()['groups'])
        except ValueError as exc:
            interpretation._restore(previous)
            interpretation._history = history
            messagebox.showinfo("Interpretação", str(exc))
            return
        refresh_interpretation()

    def merge_optical_selected():
        selected = list(optical_group_list.curselection())
        if len(selected) < 2:
            messagebox.showinfo("Unir clusters", "Selecione pelo menos dois clusters.")
            return
        apply_interpretation("merge_clusters", selected)

    def undo_optical_groups():
        apply_interpretation("undo")

    def reset_optical_groups():
        apply_interpretation("reset_clusters")

    def create_family_selected():
        selected = list(optical_group_list.curselection())
        if not selected:
            messagebox.showinfo("Criar família", "Selecione um ou mais clusters acima.")
            return
        name = simpledialog.askstring("Criar família", "Nome da família (opcional):", parent=root)
        if name is not None:
            apply_interpretation("create_family", selected, name.strip())

    def select_legacy_mask():
        prepared = session.get('prepared')
        selected = family_group_list.curselection()
        if prepared is None or not selected:
            return
        members = set().union(*(set(prepared['interpretation'].families[i]['clusters']) for i in selected))
        state = prepared['review']
        chosen = [c for c in state.class_ids if state.groups[c] <= members]
        if not chosen:
            return
        session['workspace_selected_groups'] = chosen
        state.selected_cluster = chosen[0]
        review_display_var.set('Máscara selecionada')
        isolate_var.set(True)
        refresh_review_controls()
        refresh_group_listboxes()
        session['compare_cache'] = None
        rebuild_compare_sources()
        change_preview()

    def change_family_membership(method):
        families = list(family_group_list.curselection())
        clusters = list(optical_group_list.curselection())
        if len(families) != 1 or not clusters:
            messagebox.showinfo("Família", "Selecione uma família e um ou mais clusters acima.")
            return
        apply_interpretation(method, families[0], clusters)

    def merge_family_selected():
        selected = list(family_group_list.curselection())
        if len(selected) < 2:
            messagebox.showinfo("Unir famílias", "Selecione pelo menos duas famílias.")
            return
        apply_interpretation("merge_families", selected)

    def delete_family_selected():
        selected = list(family_group_list.curselection())
        if len(selected) != 1:
            messagebox.showinfo("Excluir família", "Selecione exatamente uma família.")
            return
        apply_interpretation("delete_family", selected[0])

    def rename_selected_group(listbox, kind):
        selected = list(listbox.curselection())
        if len(selected) != 1:
            messagebox.showinfo("Renomear", "Selecione exatamente um item para renomear.")
            return
        index = selected[0]
        interpretation = current_interpretation()
        current = interpretation.names.get(index, "") if kind == "cluster" else interpretation.families[index].get("name", "")
        name = simpledialog.askstring("Renomear", "Novo nome (deixe em branco para remover):", initialvalue=current, parent=root)
        if name is not None:
            apply_interpretation(f"rename_{kind}", index, name.strip())

    def recolor_selected_group(listbox, kind):
        selected = list(listbox.curselection())
        if len(selected) != 1:
            messagebox.showinfo("Cor", "Selecione exatamente um item para trocar a cor.")
            return
        _rgb_float, rgb_hex = colorchooser.askcolor(parent=root, title="Escolher cor")
        if rgb_hex is not None:
            rgb = tuple(int(rgb_hex[i:i + 2], 16) for i in (1, 3, 5))
            apply_interpretation(f"recolor_{kind}", selected[0], rgb)

    def _draw_metric_chart(canvas, k_values, values, title, suggested_k):
        """Redraw one small line chart from already-computed metric values.

        Pure presentation: never recomputes anything, so binding this to a Canvas
        <Configure> event (window/pane resize) is safe and free of scientific cost.
        """
        canvas.delete("all")
        width = max(1, canvas.winfo_width())
        height = max(1, canvas.winfo_height())
        canvas.create_text(6, 4, anchor="nw", text=title, font=theme.font('chart_title', bold=True), fill=theme.TEXT_ON_CANVAS)
        pad_left, pad_right, pad_top, pad_bottom = 34, 12, 20, 18
        plot_w = max(1, width - pad_left - pad_right)
        plot_h = max(1, height - pad_top - pad_bottom)
        finite = [(k, v) for k, v in zip(k_values, values) if v == v]  # v == v is the NaN check
        if len(finite) < 2:
            canvas.create_text(width / 2, height / 2, text="dados insuficientes", fill=theme.CANVAS_MUTED, font=theme.font('meta'))
            return
        ks = [k for k, _ in finite]
        vs = [v for _, v in finite]
        kmin, kmax = min(ks), max(ks)
        vmin, vmax = min(vs), max(vs)
        if vmax - vmin < 1e-12:
            vmin, vmax = vmin - 1.0, vmax + 1.0

        def to_xy(k, v):
            fx = 0.5 if kmax == kmin else (k - kmin) / (kmax - kmin)
            fy = (v - vmin) / (vmax - vmin)
            return pad_left + fx * plot_w, pad_top + (1.0 - fy) * plot_h

        canvas.create_line(pad_left, pad_top, pad_left, pad_top + plot_h, fill=theme.CANVAS_MUTED)
        canvas.create_line(pad_left, pad_top + plot_h, pad_left + plot_w, pad_top + plot_h, fill=theme.CANVAS_MUTED)
        points = [to_xy(k, v) for k, v in finite]
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            canvas.create_line(x0, y0, x1, y1, fill=theme.CANVAS_ACCENT, width=2)
        for (k, _v), (x, y) in zip(finite, points):
            is_suggested = suggested_k is not None and k == suggested_k
            r = 4.5 if is_suggested else 2.5
            fill = theme.CANVAS_ACCENT if is_suggested else theme.TEXT_ON_CANVAS
            canvas.create_oval(x - r, y - r, x + r, y + r, fill=fill, outline=theme.CANVAS_ACCENT)
        # Every evaluated K gets its own X-axis label, not just the range endpoints, so
        # the reader never has to guess which points fall between kmin and kmax.
        for k in ks:
            x, _ = to_xy(k, vmin)
            canvas.create_text(x, pad_top + plot_h + 3, anchor="n", text=str(k), font=theme.font('chart_tick'), fill=theme.CANVAS_MUTED)

    def open_cluster_count_diagnostics():
        """On-demand only (never runs during a normal analysis): evaluates a small range
        of candidate optical-cluster counts on the SAME cached k-means sample already
        computed by the real analysis. Decision support only: never claims a single
        "correct" K (the suggested K is a statistical hint, not a mineralogical one), and
        never re-runs or replaces the current analysis by itself."""
        prepared = session.get("prepared")
        if prepared is None:
            messagebox.showinfo("Diagnóstico", "Execute uma análise antes de avaliar o intervalo de K.")
            return
        sample_X = prepared["optical"].get("sample_X")
        if sample_X is None:
            messagebox.showinfo("Diagnóstico", "Amostra do k-means indisponível para esta análise.")
            return
        current_k = int(prepared["optical"]["clusters"])
        k_values = sorted({k for k in range(max(2, current_k - 3), min(12, current_k + 3) + 1)})

        win = tk.Toplevel(root)
        win.title("Diagnóstico: intervalo de clusters ópticos")
        win.geometry("680x680")
        win.minsize(560, 480)
        ttk.Label(
            win,
            text=(
                "Sugestão baseada no maior índice de silhueta do intervalo avaliado. "
                "A escolha final depende da interpretação petrográfica."
            ),
            wraplength=640, foreground=theme.TEXT_SECONDARY, padding=(10, 10, 10, 4),
        ).pack(anchor="w")
        suggested_var = tk.StringVar(value="Calculando…")
        win._suggested_var = suggested_var  # keep alive: Tk deletes the one-shot after()
        # callback (and its closure) right after it fires, which would otherwise let this
        # StringVar be garbage-collected and unset its Tcl variable (same pitfall as
        # PhotoImage retention elsewhere in this file).
        ttk.Label(win, textvariable=suggested_var, padding=(10, 0, 10, 4), font=theme.font('body', bold=True)).pack(anchor="w")

        diag_state = {"result": None, "suggested_k": None}
        chart_specs = [
            ("inertia", "Inércia vs K"),
            ("silhouette", "Silhueta vs K"),
            ("calinski_harabasz", "Calinski-Harabasz vs K"),
            ("davies_bouldin", "Davies-Bouldin vs K"),
        ]
        charts_frame = ttk.Frame(win, padding=(10, 0, 10, 4))
        charts_frame.pack(fill="both", expand=True)
        for col in (0, 1):
            charts_frame.columnconfigure(col, weight=1)
        for row in (0, 1):
            charts_frame.rowconfigure(row, weight=1)
        charts = {}

        def redraw_charts(_event=None):
            result = diag_state["result"]
            if result is None:
                return
            for key, (canvas, title) in charts.items():
                values = result[key]
                if key != "inertia":
                    values = [value if valid else float("nan") for value, valid in zip(
                        values, result.get("diagnostic_valid", [True] * len(values)))]
                _draw_metric_chart(canvas, result["k_values"], values, title, diag_state["suggested_k"])

        for i, (key, title) in enumerate(chart_specs):
            c = tk.Canvas(charts_frame, bg=theme.BG_CANVAS, highlightthickness=0, bd=0, height=130)
            c.grid(row=i // 2, column=i % 2, sticky="nsew", padx=4, pady=4)
            c.bind("<Configure>", redraw_charts)
            charts[key] = (c, title)

        status_lbl = ttk.Label(win, text="Calculando…", padding=(10, 0))
        status_lbl.pack(anchor="w")

        columns = ("k", "inertia", "silhouette", "calinski_harabasz", "davies_bouldin")
        headers = {
            "k": "K", "inertia": "Inércia", "silhouette": "Silhueta (amostrada)",
            "calinski_harabasz": "Calinski-Harabasz", "davies_bouldin": "Davies-Bouldin",
        }
        tree = ttk.Treeview(win, columns=columns, show="headings", height=min(len(k_values), 6))
        tree.tag_configure("suggested", background=theme.SUGGESTED, foreground=theme.TEXT_ON_ACCENT)
        for c in columns:
            tree.heading(c, text=headers[c])
            tree.column(c, width=100, anchor="center")
        tree.pack(fill="both", expand=True, padx=10, pady=(4, 4))

        def fmt(v):
            return "n/d" if v != v else f"{v:.3f}"  # v != v is the NaN check, no extra import needed

        def use_selected_k():
            sel = tree.selection()
            if not sel:
                return
            chosen_k = int(tree.item(sel[0], "values")[0])
            clusters_var.set(chosen_k)
            status_var.set(f"K={chosen_k} definido. Clique em Gerar classificação para aplicar.")
            win.destroy()

        def use_suggested_k():
            suggested_k = diag_state["suggested_k"]
            if suggested_k is None:
                return
            clusters_var.set(suggested_k)
            status_var.set(f"K={suggested_k} definido. Clique em Gerar classificação para aplicar.")
            win.destroy()

        btns = ttk.Frame(win, padding=(10, 0, 10, 10))
        btns.pack(fill="x")
        ttk.Button(btns, text="Usar K sugerido", command=use_suggested_k).pack(side="left")
        ttk.Button(btns, text="Usar K selecionado", command=use_selected_k).pack(side="left", padx=(6, 0))
        ttk.Button(btns, text="Fechar", command=win.destroy).pack(side="right")

        def run_evaluation():
            result = core.evaluate_cluster_counts(sample_X, k_values, seed=13)
            validity = result.get("diagnostic_valid", [True] * len(result["k_values"]))
            suggested_k = core.recommend_cluster_count(
                result["k_values"], result["silhouette"], diagnostic_valid=validity)
            diag_state["result"] = result
            diag_state["suggested_k"] = suggested_k
            for i, k in enumerate(result["k_values"]):
                tags = ("suggested",) if k == suggested_k else ()
                tree.insert("", "end", values=(
                    k, fmt(result["inertia"][i]),
                    *(fmt(result[key][i]) if validity[i] else "n/d" for key in (
                        "silhouette", "calinski_harabasz", "davies_bouldin")),
                ), tags=tags)
            if suggested_k is not None:
                suggested_var.set(f"K = {suggested_k} (sugerido)")
            else:
                suggested_var.set("Nenhum K do intervalo teve silhueta válida; sem sugestão.")
            text = f"Amostra: {len(sample_X)} pixels padronizados (a mesma usada na análise atual)."
            invalid_k = [str(k) for k, valid in zip(result["k_values"], validity) if not valid]
            if invalid_k:
                text += (f" Agrupamentos degenerados (K={', '.join(invalid_k)}): "
                         "índices de separação não interpretáveis (n/d).")
            status_lbl.config(text=text)
            redraw_charts()

        win.after(50, run_evaluation)  # let the window paint "Calculando…" before the (short) computation

    def analyze():
        images = session.get("images", [])
        if not images:
            messagebox.showwarning("Entrada incompleta", "Selecione uma ou mais imagens.")
            return
        if not out_var.get():
            out_var.set(str(Path(images[0]["path"]).parent / "riima_output"))
        active_flags = [v.get() for v in session.get("_active_vars", [])]
        try:
            selected_paths, ref_idx = core.normalize_image_selection(
                [im["path"] for im in images], active_flags, int(reference_var.get())
            )
        except ValueError as exc:
            messagebox.showwarning("Seleção de imagens", str(exc))
            return
        analyze_btn.config(state="disabled")
        if session.get('prepared') is not None and not save_project_current():
            analyze_btn.config(state='normal')
            return
        for timer in ('review_after', 'autosave_after'):
            if session.get(timer) is not None:
                root.after_cancel(session[timer])
                session[timer] = None
        start = time.perf_counter()
        try:
            optical_cfg = core.Config(clusters=int(clusters_var.get()))
            mask_mode = _MASK_MODE_LABELS.get(mask_mode_var.get(), "auto")
            saved = None
            try:
                with export_batch():
                    prepared = core.prepare_analysis(
                        selected_paths, out_var.get(), reference_index=ref_idx,
                        optical_cfg=optical_cfg, stage_callback=set_status,
                        mask_mode=mask_mode,
                    )
                    # Replace the workspace only after preparation succeeds. An invalid
                    # input, unreadable file or registration failure keeps the last result.
                    summary_var.set("")
                    session["compare_mode"] = False
                    session["compare_cache"] = None
                    view_type_var.set("layers")
                    compare_panel.pack_forget()
                    layers_panel.pack_forget()
                    session["viewport"] = None
                    session["prepared"] = prepared
                    session['project_path'] = str(Path(prepared['output']) / 'projeto.riima')
                    review_enabled_var.set(False)
                    review_display_var.set('Sobreposição')
                    isolate_var.set(False)
                    contours_var.set(False)
                    review_tool_var.set('pan')
                    setup_review()
                    refresh_group_listboxes()

                    status_map = {}
                    for entry in prepared["optical"].get("registered_images", []):
                        reg = entry.get("registration")
                        if reg is None:
                            status_map[entry['name']] = 'imagem incluída'
                            continue
                        status_map[entry["name"]] = (
                            "referência" if reg.get("is_reference")
                            else "Confira o alinhamento: coordenadas preservadas; registro não verificado."
                            if reg.get('registration_warning')
                            else core.registration_quality_label(reg["score_before"], reg["score_after"])
                        )
                    update_row_statuses(status_map)

                    rebuild_layers(reset=True)
                    rebuild_compare_sources()
                    update_summary_text()
                    saved = autosave_current()
            except OSError as exc:
                if saved is None:
                    raise
                if saved:
                    messagebox.showerror("Salvar resultados", str(exc))
                saved = False
            session["analysis_seconds"] = time.perf_counter() - start
            update_context_controls()
            refresh_preview_content(legend=True)
            overlap_note = ""
            if prepared["optical"].get("overlap_status") == "warn":
                overlap_note = " Atenção: sobreposição comum entre as imagens está baixa."
            mask_note = ""
            if mask_mode == "auto":
                overlap_mask = prepared["optical"].get("overlap")
                valid_mask = prepared["optical"].get("valid")
                if overlap_mask is not None and valid_mask is not None and overlap_mask.mean() > 0:
                    coverage = float(valid_mask.mean()) / float(overlap_mask.mean())
                    if coverage < _MASK_COVERAGE_WARN_RATIO:
                        mask_note = (
                            " Atenção: a máscara automática excluiu uma fração grande da "
                            "área registrada; compare a classificação com a imagem original e considere "
                            "\"Somente sobreposição comum\" em Classificação."
                        )
            status_var.set(
                (
                    f"Análise concluída em {session['analysis_seconds']:.1f} s com {len(selected_paths)} imagem(ns)."
                    f"{overlap_note}{mask_note} "
                    + ("Resultados salvos automaticamente." if saved else "Salvamento pendente; use Salvar projeto.")
                ).replace('.', ',')
            )
        except core.InsufficientOverlapError as exc:
            if session.get('prepared') is None:
                session["current_rgb"] = None
                rebuild_layers(reset=True)
                refresh_preview_content(legend=True)
            status_var.set("Sobreposição insuficiente entre as imagens.")
            messagebox.showerror("Sobreposição insuficiente", str(exc))
        except Exception as exc:
            if session.get('prepared') is None:
                session["current_rgb"] = None
                rebuild_layers(reset=True)
                refresh_preview_content(legend=True)
            status_var.set("Erro na análise.")
            messagebox.showerror("Erro", str(exc))
        finally:
            analyze_btn.config(state="normal")

    def save_png_export(rgb, path, title):
        """Complete a PNG before replacing an existing user-selected export."""
        import os
        import tempfile
        temporary = None
        try:
            with export_batch():
                destination = staged_file(path)
                with tempfile.NamedTemporaryFile(prefix='.riima-png-', suffix='.png',
                                                 dir=destination.parent, delete=False) as file:
                    temporary = Path(file.name)
                Image.fromarray(rgb).save(temporary, format='PNG', compress_level=3, optimize=False)
                os.replace(temporary, destination)
            return True
        except Exception as exc:
            messagebox.showerror(title, str(exc))
            status_var.set(f'Não foi possível exportar o PNG: {exc}')
            return False
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def export_current():
        prepared = prepared_for_view()
        rgb = session.get("current_rgb")
        if prepared is None or rgb is None:
            messagebox.showinfo("Exportar", "Execute uma análise antes de exportar uma visualização.")
            return
        if session.get("compare_mode"):
            label = f"comparacao_{split_left_var.get()}_{split_right_var.get()}"
        else:
            label = review_display_var.get()
        suggested = label.lower().replace(" | ", "_").replace(" ", "_").replace("ó", "o").replace("ã", "a").replace("í", "i").replace("ç", "c") + ".png"
        path = filedialog.asksaveasfilename(
            defaultextension=".png", initialfile=suggested, filetypes=[("PNG", "*.png")],
        )
        if not path:
            return
        if save_png_export(rgb, path, 'Exportar visualização'):
            status_var.set(f"Visualização exportada: {path}")

    def export_selected_mask():
        prepared = prepared_for_view()
        if prepared is None:
            return
        chosen = selected_workspace_groups()
        if not chosen:
            messagebox.showinfo('Exportar máscara', 'Selecione um ou mais grupos em Seleção e Agrupamento.')
            return
        path = filedialog.asksaveasfilename(defaultextension='.png',
                                           initialfile='mascara_grupos_' + '_'.join(map(str, chosen)) + '.png',
                                           filetypes=[('Máscara PNG', '*.png')])
        if not path:
            return
        rgb = viewer.render_workspace(prepared, 'Máscara selecionada', selected_groups=chosen)
        if save_png_export(rgb, path, 'Exportar máscara'):
            status_var.set(f'Máscara da imagem atual exportada: {path}')

    def export_light_metrics():
        prepared = session.get('prepared')
        if prepared is None:
            return
        if not prepare_output_destination():
            return
        set_status('Calculando métricas das máscaras e uma amostra de atributos...')
        try:
            directory = metrics.export_metrics(prepared)
            status_var.set(f'Métricas exportadas em CSV: {directory}')
        except Exception as exc:
            messagebox.showinfo('Exportar métricas', str(exc))
            status_var.set(str(exc))

    def export_descriptive_report():
        prepared = session.get('prepared')
        if prepared is None:
            return
        if prepared['review'].has_pending:
            messagebox.showinfo('Exportar relatório', 'Atualize a classificação antes de exportar o relatório.')
            return
        path = filedialog.asksaveasfilename(
            parent=root, title='Exportar relatório descritivo', defaultextension='.html',
            initialdir=prepared['output'], initialfile='relatorio_analise_descritiva.html',
            filetypes=[('Relatório HTML', '*.html')])
        if not path:
            return
        if not prepare_output_destination():
            return
        try:
            from . import report as riima_report
            exported = riima_report.export_report(prepared, path, stage_callback=set_status)
            status_var.set(f'Relatório descritivo exportado: {exported}')
        except Exception as exc:
            messagebox.showerror('Exportar relatório', str(exc))
            status_var.set(f'Não foi possível exportar o relatório: {exc}')

    # ---------- shell ----------
    header = ttk.Frame(root, padding=(14, 10, 14, 6))
    header.pack(fill="x")
    ttk.Label(header, text="RIIMA", style="Title.TLabel").pack(side="left")
    ttk.Label(
        header,
        text="Really Intuitive Image Mapping by Affinity",
        foreground=theme.TEXT_SECONDARY,
    ).pack(side="left", padx=14, pady=(5, 0))


    project_actions = ttk.Frame(header)
    project_actions.pack(side='right')
    ttk.Button(project_actions, text='Abrir projeto...', command=open_project).pack(side='left', padx=3)
    ttk.Button(project_actions, text='Salvar projeto', command=save_project_current).pack(side='left', padx=3)
    stepbar = ttk.Frame(root, padding=(14, 3, 14, 6))
    stepbar.pack(fill='x')
    step_buttons = {}
    for index, (key, label) in enumerate(_WORKFLOW_STEPS):
        button = ttk.Button(stepbar, text=f'{index + 1}. {label}', command=lambda step=key: go_to_step(step))
        button.pack(side='left', fill='x', expand=True, padx=3)
        step_buttons[key] = button

    main = ttk.Frame(root, padding=(10, 4, 10, 4))
    main.pack(fill="both", expand=True)
    main.columnconfigure(1, weight=1)
    main.rowconfigure(0, weight=1)

    # ---------- sidebar (scrollable: viewer stays dominant even at small window sizes) ----------
    sidebar_container = ttk.Frame(main, width=310)
    sidebar_container.grid(row=0, column=0, sticky="nsw")
    sidebar_container.grid_propagate(False)
    sidebar_container.configure(width=310)

    sidebar_canvas = tk.Canvas(sidebar_container, highlightthickness=0, bd=0, bg=theme.BG_APP)
    sidebar_scroll = ttk.Scrollbar(sidebar_container, orient="vertical", command=sidebar_canvas.yview)
    sidebar_canvas.configure(yscrollcommand=sidebar_scroll.set)
    sidebar_canvas.pack(side="left", fill="both", expand=True)
    sidebar_scroll.pack(side="right", fill="y")

    sidebar = ttk.Frame(sidebar_canvas, padding=(6, 4, 12, 4))
    sidebar_window = sidebar_canvas.create_window((0, 0), window=sidebar, anchor="nw")
    def update_sidebar_scroll(event=None):
        if event is not None and event.widget == sidebar_canvas:
            sidebar_canvas.itemconfigure(sidebar_window, width=event.width)
        bounds = sidebar_canvas.bbox('all')
        if bounds is None:
            return
        height = max(1, sidebar_canvas.winfo_height())
        content = max(height, bounds[3])
        sidebar_canvas.configure(scrollregion=(0, 0, bounds[2], content))
        offset = max(0, min(content - height, sidebar_canvas.yview()[0] * content))
        sidebar_canvas.yview_moveto(offset / content)
    sidebar.bind('<Configure>', update_sidebar_scroll)
    sidebar_canvas.bind('<Configure>', update_sidebar_scroll)

    def _sidebar_wheel(event):
        region = sidebar_canvas.cget('scrollregion')
        if not region:
            return 'break'
        bounds = [float(value) for value in str(region).split()]
        content = bounds[3] - bounds[1]
        height = max(1, sidebar_canvas.winfo_height())
        if content <= height:
            sidebar_canvas.yview_moveto(0)
            return 'break'
        delta = getattr(event, "delta", 0)
        if delta == 0:
            delta = 120 if getattr(event, "num", 5) == 4 else -120
        offset = sidebar_canvas.yview()[0] * content + (-36 if delta > 0 else 36)
        sidebar_canvas.yview_moveto(max(0, min(content - height, offset)) / content)
        return 'break'

    # Mousewheel only scrolls the sidebar while the cursor is actually over it: bound
    # globally on enter and released on leave, so it never steals the preview canvas's
    # own wheel-zoom binding when the cursor is elsewhere.
    sidebar_container.bind("<Enter>", lambda _e: sidebar_canvas.bind_all("<MouseWheel>", _sidebar_wheel))
    sidebar_container.bind("<Leave>", lambda _e: sidebar_canvas.unbind_all("<MouseWheel>"))


    ttk.Label(sidebar, textvariable=workflow_hint_var, wraplength=265,
              foreground=theme.TEXT_SECONDARY).pack(anchor='w', pady=(0, 8))
    navigation = ttk.Frame(sidebar)
    navigation.pack(fill='x')
    back_btn = ttk.Button(navigation, text='Anterior', command=previous_step)
    back_btn.pack(side='left')
    next_btn = ttk.Button(navigation, text='Próximo', command=advance_step, style='Primary.TButton')
    next_btn.pack(side='right')
    step_frames = {key: ttk.Frame(sidebar) for key, _ in _WORKFLOW_STEPS}
    images_step = step_frames['images']
    classify_step = step_frames['classify']
    organize_step = step_frames['organize']
    refine_step = step_frames['refine']
    results_step = step_frames['results']
    results_status_var = tk.StringVar()
    interpretation_slot = organize_step  # Compatibility for the existing callbacks.
    ttk.Label(images_step, text="Imagens", style="Section.TLabel").pack(anchor="w", pady=(0, 6))
    ttk.Button(images_step, text="Selecionar imagens...", command=choose_images).pack(fill="x", pady=(0, 4))
    ttk.Button(images_step, text='Abrir projeto...', command=open_project).pack(fill='x', pady=(0, 4))
    images_list_frame = ttk.Frame(images_step)
    images_list_frame.pack(fill="x", pady=(0, 2))
    ttk.Label(images_list_frame, text="Nenhuma imagem selecionada.", foreground=theme.TEXT_SECONDARY).pack(anchor="w")
    ttk.Label(
        images_step,
        text="Selecione imagens do mesmo campo da lâmina. Elas serão alinhadas à referência e analisadas em conjunto. Para analisar uma imagem sozinha, selecione apenas ela.",
        wraplength=270, foreground=theme.TEXT_SECONDARY,
    ).pack(anchor="w", pady=(2, 8))

    ttk.Label(images_step, text="Pasta de resultados", foreground=theme.TEXT_SECONDARY).pack(anchor="w", pady=(0, 2))
    outrow = ttk.Frame(images_step)
    outrow.pack(fill="x")
    ttk.Entry(outrow, textvariable=out_var, width=24).pack(side="left", fill="x", expand=True, padx=(0, 5))
    ttk.Button(outrow, text="…", width=3, command=choose_out).pack(side="right")
    ttk.Checkbutton(images_step, text='Pasta automática para novas imagens',
                    variable=automatic_output_var).pack(anchor='w', pady=(4, 0))

    ttk.Separator(images_step, orient="horizontal").pack(fill="x", pady=12)
    ttk.Label(classify_step, text="Classificação", style="Section.TLabel").pack(anchor="w", pady=(0, 6))
    ttk.Label(
        classify_step,
        text="Defina K e a máscara antes de executar a análise.",
        wraplength=270, foreground=theme.TEXT_SECONDARY,
    ).pack(anchor="w", pady=(0, 6))
    clusters_row = ttk.Frame(classify_step); clusters_row.pack(fill="x", pady=(0, 4))
    ttk.Label(clusters_row, text="Clusters ópticos").pack(side="left")
    ttk.Spinbox(clusters_row, from_=2, to=12, textvariable=clusters_var, width=6).pack(side="right")
    mask_row = ttk.Frame(classify_step); mask_row.pack(fill="x", pady=(0, 4))
    ttk.Label(mask_row, text="Máscara de análise").pack(side="left")
    mask_combo = ttk.Combobox(
        mask_row, textvariable=mask_mode_var, state="readonly", width=22,
        values=list(_MASK_MODE_LABELS.keys()),
    )
    mask_combo.pack(side="right")
    ttk.Label(
        classify_step,
        text="Automática remove fundo e anotações. Use Somente sobreposição comum se alguma área válida for excluída.",
        wraplength=270, foreground=theme.TEXT_SECONDARY,
    ).pack(anchor="w", pady=(0, 8))

    analyze_btn = ttk.Button(classify_step, text="Gerar classificação", style="Primary.TButton", command=analyze)
    analyze_btn.pack(fill="x", pady=(2, 6))
    ttk.Label(
        classify_step,
        text="Os produtos da análise são salvos automaticamente.",
        wraplength=270, foreground=theme.TEXT_SECONDARY,
    ).pack(anchor="w", pady=(0, 4))

    # A stable, always-packed slot: diagnostics_frame is shown/hidden inside it (see
    # update_context_controls), which keeps this section's position fixed regardless of
    # when that toggling first happens (plain classify_step.pack()/pack_forget() cycling would
    # otherwise re-append the frame after every already-packed classify_step section, including
    # "4. Resultados", the first time it is shown).
    diagnostics_slot = ttk.Frame(classify_step)
    diagnostics_slot.pack(fill="x")
    # Deliberately compact (no LabelFrame/border): the diagnostic opens its own window, so
    # this only needs to be one row, not a tall boxed section that pushes "3.
    # Interpretação" out of view in a normal-height window.
    diagnostics_frame = ttk.Frame(diagnostics_slot, padding=(0, 4))
    ttk.Button(diagnostics_frame, text="Diagnóstico de K...", command=open_cluster_count_diagnostics).pack(fill="x")
    ttk.Label(
        diagnostics_frame,
        text="Compara separação estatística para diferentes valores de K.",
        wraplength=250, foreground=theme.TEXT_SECONDARY,
    ).pack(anchor="w", pady=(3, 0))

    ttk.Separator(classify_step, orient="horizontal").pack(fill="x", pady=12)
    optical_group_frame = ttk.LabelFrame(organize_step, text="Grupos", padding=theme.SPACE_SM)
    ttk.Label(
        optical_group_frame,
        text="Clique para selecionar um grupo. Shift + clique adiciona ou remove grupos para inspecionar a união.",
        wraplength=250, foreground=theme.TEXT_SECONDARY,
    ).pack(anchor="w", pady=(0, 5))
    optical_group_list = tk.Listbox(optical_group_frame, height=5, selectmode=tk.BROWSE, exportselection=False)
    optical_group_list.pack(fill="x")
    optical_group_list.bind("<<ListboxSelect>>", on_group_selection)
    optical_group_list.bind('<Button-1>', select_list_item)
    optical_group_list.bind('<B1-Motion>', lambda event: 'break' if event.state & 1 else select_list_item(event))
    ttk.Button(optical_group_frame, text='Limpar seleção', command=clear_group_selection).pack(fill='x', pady=(4, 0))
    ttk.Button(optical_group_frame, text='Enviar seleção para fundo',
               command=lambda: selected_groups_background(True)).pack(fill='x', pady=(4, 0))
    ttk.Button(optical_group_frame, text='Restaurar seleção do fundo',
               command=lambda: selected_groups_background(False)).pack(fill='x', pady=(4, 0))
    og_buttons = ttk.Frame(optical_group_frame); og_buttons.pack(fill="x", pady=(5, 0))
    ttk.Button(og_buttons, text="Unir", command=merge_optical_selected).pack(side="left")
    ttk.Button(og_buttons, text="Desfazer", command=undo_optical_groups).pack(side="left", padx=4)
    ttk.Button(og_buttons, text="Resetar", command=reset_optical_groups).pack(side="left")
    og_buttons2 = ttk.Frame(optical_group_frame); og_buttons2.pack(fill="x", pady=(4, 0))
    ttk.Button(og_buttons2, text="Renomear…", command=lambda: rename_selected_group(optical_group_list, "cluster")).pack(side="left")
    ttk.Button(og_buttons2, text="Cor…", command=lambda: recolor_selected_group(optical_group_list, "cluster")).pack(side="left", padx=4)

    optical_group_frame.pack(fill="x")
    review_frame = ttk.LabelFrame(refine_step, text='Revisão dos grupos', padding=theme.SPACE_SM)
    ttk.Label(review_frame, text='Revise um grupo de cada vez. Marque exemplos e atualize quando terminar.',
              wraplength=250, foreground=theme.TEXT_SECONDARY).pack(anchor='w', pady=(0, 4))
    review_cluster_combo = ttk.Combobox(review_frame, textvariable=review_cluster_var, state='readonly')
    review_cluster_combo.pack(fill='x', pady=3)
    review_cluster_combo.bind('<<ComboboxSelected>>', select_review_cluster)
    tools_row = ttk.Frame(review_frame)
    tools_row.pack(fill='x', pady=4)
    def activate_review_tool():
        review_enabled_var.set(True)
        contours_var.set(False)
        if session.get('compare_mode'):
            view_type_var.set('layers')
            on_view_type_change()
        refresh_review_controls()
        change_preview()
    tools_row.columnconfigure(0, weight=1)
    tools_row.columnconfigure(1, weight=1)
    def choose_review_tool(tool):
        review_tool_var.set(tool)
        activate_review_tool()
    for index, (text, value) in enumerate([('Manter', 'positive'), ('Remover', 'negative'),
                                          ('Navegar', 'pan'), ('Apagar', 'erase'),
                                          ('Fundo', 'background'), ('Restaurar área', 'restore')]):
        button = ttk.Button(tools_row, text=text, command=lambda tool=value: choose_review_tool(tool))
        button.grid(row=index // 2, column=index % 2, sticky='ew', pady=2, padx=(0, 2))
        review_tool_buttons[value] = button
    ttk.Label(review_frame, text='Remover exclui deste grupo. Fundo exclui de todos. Restaurar área devolve o fundo à análise. Apagar limpa marcações.',
              wraplength=250, foreground=theme.TEXT_SECONDARY).pack(anchor='w', pady=(0, 4))
    radius_row = ttk.Frame(review_frame)
    radius_row.pack(fill='x')
    ttk.Label(radius_row, text='Raio do pincel (pixels)').pack(side='left')
    ttk.Spinbox(radius_row, from_=1, to=100, textvariable=review_radius_var, width=5).pack(side='right')
    review_update_btn = ttk.Button(review_frame, text='Atualizar classificação', command=refine_review_cluster)
    review_update_btn.pack(fill='x', pady=(3, 0))
    review_undo_btn = ttk.Button(review_frame, text='Desfazer pincelada', command=undo_review_stroke)
    review_undo_btn.pack(fill='x', pady=(3, 0))
    review_reset_btn = ttk.Button(review_frame, text='Resetar grupo', command=reset_review_group)
    review_reset_btn.pack(fill='x', pady=(3, 0))
    ttk.Checkbutton(review_frame, text='Mostrar marcações', variable=show_applied_var,
                    command=change_preview).pack(anchor='w', pady=(4, 0))
    approval_row = ttk.Frame(review_frame)
    approval_row.pack(fill='x', pady=4)
    review_approve_btn = ttk.Button(approval_row, text='Grupo pronto', command=approve_review_cluster)
    review_approve_btn.pack(side='left')
    review_reopen_btn = ttk.Button(approval_row, text='Reabrir', command=reopen_review_cluster)
    review_reopen_btn.pack(side='left', padx=5)
    ttk.Label(review_frame, textvariable=review_status_var, wraplength=250).pack(anchor='w')
    ttk.Label(review_frame, text='As marcações ficam pendentes até Atualizar classificação. Grupo pronto protege toda a seleção. Salve o projeto para guardar o progresso.',
              wraplength=250, foreground=theme.TEXT_SECONDARY).pack(anchor='w', pady=(4, 0))

    legacy_masks_toggle = ttk.Checkbutton(organize_step, text='Seleções de projetos antigos',
                                         variable=legacy_masks_var, command=change_preview)
    family_group_frame = ttk.LabelFrame(organize_step, text="Associações salvas", padding=theme.SPACE_SM)
    ttk.Label(family_group_frame, text="Use as associações existentes para selecionar máscaras dos grupos atuais.", wraplength=250, foreground=theme.TEXT_SECONDARY).pack(anchor="w", pady=(0, 5))
    family_group_list = tk.Listbox(family_group_frame, height=4, selectmode=tk.BROWSE, exportselection=False)
    family_group_list.pack(fill="x", pady=(5, 0))
    family_group_list.bind('<Button-1>', select_list_item)
    family_group_list.bind('<B1-Motion>', lambda event: 'break' if event.state & 1 else select_list_item(event))
    ttk.Button(family_group_frame, text='Ver máscara', command=select_legacy_mask).pack(fill='x', pady=3)
    fg_members = ttk.Frame(family_group_frame); fg_members.pack(fill="x", pady=(5, 0))
    ttk.Button(fg_members, text="Adicionar clusters", command=lambda: change_family_membership("add_to_family")).pack(side="left")
    ttk.Button(fg_members, text="Remover clusters", command=lambda: change_family_membership("remove_from_family")).pack(side="left", padx=4)
    fg_buttons = ttk.Frame(family_group_frame); fg_buttons.pack(fill="x", pady=(5, 0))
    ttk.Button(fg_buttons, text="Unir", command=merge_family_selected).pack(side="left")
    ttk.Button(fg_buttons, text="Remover associação", command=delete_family_selected).pack(side="left", padx=4)
    ttk.Button(fg_buttons, text="Desfazer", command=undo_optical_groups).pack(side="left")
    fg_buttons2 = ttk.Frame(family_group_frame); fg_buttons2.pack(fill="x", pady=(4, 0))
    ttk.Button(fg_buttons2, text="Renomear…", command=lambda: rename_selected_group(family_group_list, "family")).pack(side="left")
    ttk.Button(fg_buttons2, text="Cor…", command=lambda: recolor_selected_group(family_group_list, "family")).pack(side="left", padx=4)


    review_frame.pack(fill='x')
    ttk.Label(results_step, text='Resultados', style='Section.TLabel').pack(anchor='w', pady=(0, 8))
    ttk.Label(results_step, textvariable=results_status_var, wraplength=260).pack(anchor='w', pady=(0, 12))
    ttk.Button(results_step, text='Salvar projeto', command=save_project_current).pack(fill='x', pady=3)
    ttk.Button(results_step, text='Salvar projeto como...', command=lambda: save_project_current(choose=True)).pack(fill='x', pady=3)
    ttk.Button(results_step, text='Exportar classificação e tabelas', command=autosave_current).pack(fill='x', pady=3)
    ttk.Button(results_step, text='Exportar máscara selecionada', command=export_selected_mask).pack(fill='x', pady=3)
    ttk.Button(results_step, text='Exportar visualização atual', command=export_current).pack(fill='x', pady=3)
    ttk.Button(results_step, text='Exportar métricas', command=export_light_metrics).pack(fill='x', pady=3)
    ttk.Label(results_step, text='Área, forma, direção e atributos em CSV. Calculados somente ao exportar.',
              wraplength=250, foreground=theme.TEXT_SECONDARY).pack(anchor='w', pady=4)
    ttk.Button(results_step, text='Exportar relatório', command=export_descriptive_report).pack(fill='x', pady=3)
    ttk.Label(results_step, text='Relatório HTML com a classificação, gráficos e tabelas da análise atual.',
              wraplength=250, foreground=theme.TEXT_SECONDARY).pack(anchor='w', pady=4)
    ttk.Button(results_step, text='Abrir pasta de resultados', command=open_output_folder).pack(fill='x', pady=3)

    # ---------- central preview + fixed legend viewport ----------
    workspace = ttk.Frame(main)
    workspace.grid(row=0, column=1, sticky="nsew")
    workspace.columnconfigure(0, weight=1)
    workspace.rowconfigure(1, weight=1)

    titlebar = ttk.Frame(workspace)
    titlebar.grid(row=0, column=0, sticky="ew", padx=(4, 0), pady=(2, 5))
    ttk.Label(titlebar, textvariable=preview_title_var, font=theme.font('viewer', bold=True)).pack(side="left")

    zoom_toolbar = ttk.Frame(titlebar)
    zoom_toolbar.pack(side="right")
    ttk.Button(zoom_toolbar, text="−", width=3, command=lambda: zoom_button(1 / 1.25)).pack(side="left", padx=1)
    ttk.Button(zoom_toolbar, text="+", width=3, command=lambda: zoom_button(1.25)).pack(side="left", padx=1)
    ttk.Button(zoom_toolbar, text="Ajustar", command=set_fit_mode).pack(side="left", padx=1)
    ttk.Button(zoom_toolbar, text="Largura", command=set_width_mode).pack(side="left", padx=1)
    ttk.Button(zoom_toolbar, text="100%", command=set_100_mode).pack(side="left", padx=1)

    visual = ttk.Frame(workspace)
    visual.grid(row=1, column=0, sticky="nsew")
    visual.columnconfigure(0, weight=1)
    visual.columnconfigure(1, weight=0, minsize=265)
    visual.rowconfigure(0, weight=1)

    preview_canvas = tk.Canvas(visual, bg=theme.BG_CANVAS, highlightthickness=0, bd=0)
    preview_canvas.grid(row=0, column=0, sticky="nsew", padx=(4, 8))
    preview_canvas.bind("<Configure>", schedule_redraw)
    preview_canvas.bind("<ButtonPress-1>", on_canvas_press)
    preview_canvas.bind("<B1-Motion>", on_canvas_drag)
    preview_canvas.bind("<ButtonRelease-1>", on_canvas_release)
    preview_canvas.bind("<Motion>", on_canvas_motion)
    preview_canvas.bind("<Double-Button-1>", on_canvas_double_click)
    preview_canvas.bind("<MouseWheel>", on_canvas_wheel)   # Windows/macOS
    preview_canvas.bind("<Button-4>", on_canvas_wheel)      # Linux, wheel up
    preview_canvas.bind("<Button-5>", on_canvas_wheel)      # Linux, wheel down

    legend = ttk.LabelFrame(visual, text="Legenda", padding=6, width=260, height=520)
    legend.grid(row=0, column=1, sticky="ns", padx=(0, 4))
    legend.grid_propagate(False)
    legend_canvas = tk.Canvas(legend, width=230, highlightthickness=0, bd=0, bg=theme.BG_APP)
    legend_scroll = ttk.Scrollbar(legend, orient="vertical", command=legend_canvas.yview)
    legend_canvas.configure(yscrollcommand=legend_scroll.set)
    legend_canvas.pack(side="left", fill="both", expand=True)
    legend_scroll.pack(side="right", fill="y")
    legend_rows = ttk.Frame(legend_canvas)
    legend_window = legend_canvas.create_window((0, 0), window=legend_rows, anchor="nw")
    legend_rows.bind("<Configure>", lambda _e: legend_canvas.configure(scrollregion=legend_canvas.bbox("all")))
    legend_canvas.bind("<Configure>", lambda e: legend_canvas.itemconfigure(legend_window, width=max(1, e.width - 2)))
    ttk.Label(legend_rows, text="A legenda aparece depois da análise.", wraplength=210, foreground=theme.TEXT_SECONDARY).pack(anchor="w")


    view_type_row = ttk.Frame(workspace)
    view_type_row.grid(row=2, column=0, sticky='ew', padx=4, pady=(7, 2))
    ttk.Label(view_type_row, text='Imagem').pack(side='left', padx=(0, 5))
    review_image_combo = ttk.Combobox(view_type_row, textvariable=review_image_var, state='readonly', width=29)
    review_image_combo.pack(side='left', padx=(0, 15))
    review_image_combo.bind('<<ComboboxSelected>>', select_review_image)
    def choose_view_mode(mode):
        view_type_var.set(mode)
        on_view_type_change()
    for text, mode in [('Trabalho', 'layers'), ('Comparar', 'compare')]:
        button = ttk.Button(view_type_row, text=text, command=lambda value=mode: choose_view_mode(value))
        button.pack(side='left', padx=(0, 7))
        view_mode_buttons[mode] = button
    view_controls = ttk.Frame(workspace)
    view_controls.grid(row=3, column=0, sticky='ew', padx=4, pady=(0, 4))
    display_controls = ttk.Frame(view_controls)
    display_controls.pack(fill='x', pady=4)
    review_display_combo = ttk.Combobox(display_controls, textvariable=review_display_var, state='readonly',
                                       values=['Imagem', 'Sobreposição', 'Classificação', 'Máscara selecionada'], width=22)
    review_display_combo.pack(side='left', padx=(0, 8))
    review_display_combo.bind('<<ComboboxSelected>>', lambda _e: change_preview())
    ttk.Checkbutton(display_controls, text='Destacar seleção', variable=isolate_var,
                    command=change_preview).pack(side='left', padx=(0, 8))
    ttk.Checkbutton(display_controls, text='Contornos', variable=contours_var,
                    command=change_preview).pack(side='left', padx=(0, 8))
    ttk.Label(display_controls, text='Opacidade').pack(side='left', padx=(0, 4))
    ttk.Scale(display_controls, from_=0, to=100, variable=review_opacity_var, length=100,
              command=schedule_workspace_redraw).pack(side='left')
    layers_panel = ttk.Frame(view_controls)  # No layer stack is exposed in the workspace.
    layers_list_frame = ttk.Frame(layers_panel)
    compare_panel = ttk.Frame(view_controls)
    ttk.Label(compare_panel, text="Esquerda").pack(side="left", padx=(0, 3))
    split_left_combo = ttk.Combobox(compare_panel, textvariable=split_left_var, state="readonly", width=16)
    split_left_combo.pack(side="left", padx=(0, 8))
    split_left_combo.bind("<<ComboboxSelected>>", on_compare_source_change)
    ttk.Label(compare_panel, text="Direita").pack(side="left", padx=(0, 3))
    split_right_combo = ttk.Combobox(compare_panel, textvariable=split_right_var, state="readonly", width=16)
    split_right_combo.pack(side="left")
    split_right_combo.bind("<<ComboboxSelected>>", on_compare_source_change)
    # compare_panel starts unpacked; on_view_type_change() shows/hides it vs layers_panel.

    footer = ttk.Frame(root, padding=(14, 5, 14, 10))
    footer.pack(fill="x")
    ttk.Separator(footer, orient="horizontal").pack(fill="x", pady=(0, 6))
    ttk.Label(footer, textvariable=status_var, foreground=theme.TEXT_SECONDARY).pack(anchor="w")
    ttk.Label(footer, textvariable=summary_var, style="Summary.TLabel", wraplength=1200).pack(anchor="w", pady=(2, 0))

    project_view_variables = tuple(value for value in locals().values() if isinstance(value, tk.Variable))
    project_view_buttons = (tuple(value for value in locals().values() if isinstance(value, ttk.Button))
                            + tuple(review_tool_buttons.values()) + tuple(view_mode_buttons.values())
                            + tuple(step_buttons.values()))
    update_context_controls()
    root.protocol('WM_DELETE_WINDOW', close_app)
    root.after_idle(render_preview_canvas)
    if _test_hook is not None:
        root.update_idletasks()
        _test_hook(locals())
        return
    root.mainloop()


def parse_args():
    p = argparse.ArgumentParser(description="RIIMA: lightweight optical-affinity mapping for co-registered thin-section images.")
    p.add_argument("--ppl", help="PPL image (legacy two-image form)")
    p.add_argument("--cpol", help="cPol image (legacy two-image form)")
    p.add_argument("--image", action="append", help="Image path; repeat for N images (e.g. --image a.png --image b.png)")
    p.add_argument("--reference", type=int, default=0, help="Index of the reference image among --image entries")
    p.add_argument("--output", default="riima_output", help="Output directory")
    p.add_argument("--clusters", type=int, default=8, choices=range(2, 13), metavar="2..12")
    p.add_argument("--sample", type=int, default=80_000, help="Pixels sampled for k-means fitting")
    p.add_argument("--merge", default="", help="Optional cluster merges, e.g. 0+4+7;2+5+6")
    p.add_argument("--diagnostics", action="store_true", help="Also save diagnostic-only intermediate products")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.ppl and not args.cpol and not args.image:
        gui(); return
    try:
        paths, reference_index = core.resolve_cli_images(args.ppl, args.cpol, args.image, args.reference)
    except ValueError as exc:
        raise SystemExit(str(exc))
    optical_cfg = core.Config(clusters=args.clusters, sample_pixels=args.sample)
    try:
        with export_batch():
            result = core.run_analysis(
                paths, args.output, optical_cfg=optical_cfg, reference_index=reference_index,
                diagnostics=args.diagnostics,
            )
            if args.merge.strip():
                groups = core.parse_merge_spec(args.merge, optical_cfg.clusters)
                interpretation = result["interpretation"]
                # Resolve each requested raw group against current interpreted indices.
                for group in groups:
                    selected = [i for i, members in enumerate(interpretation.groups) if members <= set(group)]
                    if len(selected) > 1:
                        interpretation.merge_clusters(selected)
                core.save_session_outputs(result)
    except core.InsufficientOverlapError as exc:
        raise SystemExit(str(exc))
    print(f"Done: {result['output']}")
    print(viewer.human_analysis_summary(result))
    if result["optical"].get("overlap_status") == "warn":
        print("Warning: common overlap across active images is low; check registration quality.")


if __name__ == "__main__":
    main()
