#!/usr/bin/env python3
"""RIIMA desktop application: state, controls, and the CLI entry point.

This is the only module that imports ``tkinter``. It calls ``riima.core`` for the
automatic analysis, ``riima.review`` for interactive learning and project persistence,
and ``riima.viewer`` for presentation/navigation.
"""
from __future__ import annotations

import argparse
from contextvars import copy_context
import gc
from queue import Empty, Queue
import tempfile
import threading
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

_WORKFLOW_STEPS = [('preparation', 'Preparação'), ('interpretation', 'Interpretação')]
_STEP_ALIASES = dict(images='preparation', classify='preparation', organize='interpretation',
                     refine='interpretation', results='interpretation')


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
    icon_cache, tooltips = {}, {}
    similarity_windows = set()

    def desktop_icon(name, selected=False):
        key = (name, selected)
        if key not in icon_cache:
            image = theme.icon_image(name, color=theme.TEXT_ON_ACCENT if selected else theme.TEXT_PRIMARY)
            icon_cache[key] = ImageTk.PhotoImage(image, master=root)
        return icon_cache[key]

    def attach_tooltip(widget, text):
        tip = dict(text=text, popup=None, after=None)
        tooltips[widget] = tip
        def hide(_event=None):
            if tip['after'] is not None:
                root.after_cancel(tip['after'])
                tip['after'] = None
            if tip['popup'] is not None:
                if tip['popup'].winfo_exists():
                    tip['popup'].destroy()
                tip['popup'] = None
        def show(_event=None):
            hide()
            if not widget.winfo_exists():
                return
            popup = tk.Toplevel(root)
            popup.overrideredirect(True)
            popup.attributes('-topmost', True)
            popup.geometry(f'+{widget.winfo_rootx()}+{widget.winfo_rooty() + widget.winfo_height() + 4}')
            tk.Label(popup, text=tip['text']() if callable(tip['text']) else tip['text'], justify='left', wraplength=330, padx=8, pady=6,
                     font=theme.FONT_META, background=theme.BG_PANEL, foreground=theme.TEXT_PRIMARY,
                     relief='solid', borderwidth=1).pack()
            tip['popup'] = popup
        def schedule(_event=None):
            hide()
            tip['after'] = root.after(450, show)
        widget.bind('<Enter>', schedule, add='+')
        for event in ('<Leave>', '<FocusOut>', '<ButtonPress>', '<Escape>'):
            widget.bind(event, hide, add='+')
        def destroy(event):
            if event.widget is widget:
                hide()
                tooltips.pop(widget, None)
        widget.bind('<Destroy>', destroy, add='+')
        widget.bind('<FocusIn>', show, add='+')
        widget.bind('<F1>', show, add='+')
        return widget

    def action_button(parent, text, command, *, icon=None, description='', **options):
        options.setdefault('width', 0)
        if icon is not None:
            options['image'] = desktop_icon(icon)
            options.setdefault('compound', 'left')
            if options['compound'] == 'image':
                options.setdefault('style', 'Icon.TButton')
        button = ttk.Button(parent, text=text, command=command, takefocus=True, **options)
        return attach_tooltip(button, text + ((' — ' if text else '') + description if description else ''))

    def context_help(parent, text):
        button = action_button(parent, 'Ajuda', lambda: None, icon='help', description=text)
        button.config(command=lambda: button.event_generate('<F1>'))
        return button

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
    workflow_var = tk.StringVar(value='preparation')
    analysis_mode_var = tk.StringVar(value='Registradas (mesmo campo)')
    active_group_var = tk.StringVar(value='Grupo ativo: —')
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
            selected = tool == review_tool_var.get()
            button.config(style='Selected.Icon.TButton' if selected else 'Icon.TButton', image=desktop_icon(tool, selected))
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
        "project_dirty": False,
        "review_after": None,
        "stroke_last": None,
        "workspace_selected_groups": [],
        "busy": False,
        "brush_position": None,
        "science_revision": 0,
        "similarity_cache": None,
        "similarity_rows": [],
        "group_selection_history": [],
        "operation_after": None,
        "operation_frame": 0,
    }
    retained_legend_rows = ()
    ui_thread = threading.get_ident()
    operation_messages = Queue()
    operation_done = tk.BooleanVar(root, False)

    def set_status(text: str):
        if threading.get_ident() != ui_thread:
            operation_messages.put(('status', text))
            return
        status_var.set(text)
        root.update_idletasks()

    def run_operation(message, function, *args, **kwargs):
        """Run one computation while Tk dispatches events with editing disabled.

        Copy the caller's context so nested export operations retain their outer
        transaction. Only the UI thread accesses widgets, including status updates.
        """
        if session['busy']:
            raise RuntimeError('Aguarde o processamento atual terminar.')
        session['busy'] = True
        preview_canvas.delete('brush_cursor')
        previous_cursor = preview_canvas.cget('cursor')
        preview_canvas.config(cursor='watch')
        controls = []

        def disable(widget):
            for child in widget.winfo_children():
                if isinstance(child, (ttk.Button, ttk.Menubutton, ttk.Entry, ttk.Combobox, ttk.Spinbox,
                                      ttk.Checkbutton, ttk.Radiobutton, ttk.Scale, ttk.Treeview)):
                    controls.append((child, child.state()))
                    child.state(['disabled'])
                elif isinstance(child, tk.Listbox):
                    controls.append((child, child.cget('state')))
                    child.config(state='disabled')
                disable(child)

        disable(root)
        set_status(message)
        start_operation_indicator()
        root.update_idletasks()
        operation_done.set(False)
        result = []
        context = copy_context()

        def work():
            try:
                operation_messages.put(('result', context.run(function, *args, **kwargs)))
            except BaseException as exc:
                operation_messages.put(('error', exc))

        def poll():
            try:
                while True:
                    kind, value = operation_messages.get_nowait()
                    if kind == 'status':
                        status_var.set(value)
                    else:
                        result.append((kind, value))
                        operation_done.set(True)
                        return
            except Empty:
                root.after(15, poll)

        # Tk variables can belong to unreachable cycles; finalize those on their
        # owning thread rather than during an allocation in the worker.
        gc_was_enabled = gc.isenabled()
        if gc_was_enabled:
            gc.collect()
            gc.disable()
        try:
            worker = threading.Thread(target=work, daemon=True)
            worker.start()
            root.after(15, poll)
            root.wait_variable(operation_done)
            worker.join()
        finally:
            poll = None
            if gc_was_enabled:
                gc.collect()
                gc.enable()
            stop_operation_indicator()
            for widget, state in controls:
                if widget.winfo_exists():
                    if isinstance(widget, tk.Listbox):
                        widget.config(state=state)
                    else:
                        widget.state(['!' + flag for flag in widget.state()])
                        widget.state(state)
            session['busy'] = False
            preview_canvas.config(cursor=previous_cursor)
            update_brush_outline()
            refresh_export_actions()
        kind, value = result[0]
        if kind == 'error':
            status_var.set('Não foi possível concluir o processamento.')
            raise value
        return value

    def selected_workspace_groups():
        prepared = session.get('prepared')
        if prepared is None or 'review' not in prepared:
            return []
        state = prepared['review']
        chosen = [c for c in session['workspace_selected_groups'] if c in state.groups]
        return chosen

    def close_similarity_windows():
        for window in tuple(similarity_windows):
            if window.winfo_exists():
                window.destroy()
            similarity_windows.discard(window)

    def refresh_similarity_windows(*, selection_only=False):
        for window in tuple(similarity_windows):
            if window.winfo_exists():
                callback = getattr(window, '_sync_selection' if selection_only else '_refresh', None)
                if callback is not None:
                    callback()

    def invalidate_similarity():
        close_similarity_windows()
        session['science_revision'] += 1
        session['similarity_cache'] = None
        session['similarity_rows'] = []
        session['project_dirty'] = True

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
        if session['busy']:
            return
        step = _STEP_ALIASES.get(step, step)
        if step not in dict(_WORKFLOW_STEPS):
            return
        if step == 'interpretation' and session.get('prepared') is None:
            return
        workflow_var.set(step)
        review_enabled_var.set(step == 'interpretation')
        review_tool_var.set('pan')
        if step == 'interpretation':
            refresh_group_listboxes()
        session['compare_cache'] = None
        rebuild_compare_sources()
        change_preview()

        refresh_similarity_windows(selection_only=True)

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
        if session['busy']:
            return
        prepared = session.get('prepared')
        if prepared is None:
            return
        state = prepared['review']
        chosen = [int(value) for value in group_tree.selection() if int(value) in state.groups]
        session['workspace_selected_groups'] = chosen
        isolate_var.set(True)
        refresh_review_controls()
        session['compare_cache'] = None
        rebuild_compare_sources()
        change_preview()
        refresh_similarity_windows(selection_only=True)

    def select_list_item(event):
        """A plain click replaces the selection; Shift toggles one additional item."""
        listbox = event.widget
        if isinstance(listbox, ttk.Treeview):
            if session['busy'] or listbox.instate(['disabled']):
                return 'break'
            row = listbox.identify_row(event.y)
            if row:
                chosen = list(listbox.selection()) if event.state & (1 | 4) else []
                if row in chosen:
                    chosen.remove(row)
                else:
                    chosen.append(row)
                listbox.selection_set(chosen)
                listbox.focus(row)
                listbox.focus_set()
                on_group_selection()
            return 'break'
        if session['busy'] or str(listbox.cget('state')) == 'disabled' or not listbox.size():
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
        # Stage inputs independently; the existing interpretation remains usable
        # until a confirmed replacement has completed successfully.
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
        go_to_step('preparation')
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
                          wraplength=216, foreground=theme.TEXT_SECONDARY).pack(anchor='w', pady=5)
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
        refresh_group_listboxes()
        refresh_review_controls()

    def review_group_label(group):
        prepared = session.get('prepared')
        return prepared['review'].group_name(group, prepared['interpretation'])

    def sync_review_groups(*, topology=True):
        prepared = session.get('prepared')
        if prepared is None or 'review' not in prepared:
            return
        state = prepared['review']
        interpretation = prepared['interpretation']
        if topology:
            state.sync_groups(interpretation.groups)
            invalidate_similarity()
        if state._catalog_edited:
            return  # Current catalog colors are independent of raw provenance.
        palette = core.cluster_palette(len(interpretation.groups))
        for i, group in enumerate(interpretation.groups):
            if state.groups.get(min(group)) == group:
                state.palette[min(group)] = interpretation.colors.get(i, palette[i])

    def refresh_review_controls():
        if session['busy']:
            return
        prepared = session.get('prepared')
        if prepared is None or 'review' not in prepared:
            return
        state = prepared['review']
        session['review_group_ids'] = state.class_ids
        from tkinter import font as tkfont
        name = review_group_label(state.selected_cluster)
        display_name = name
        label_font = tkfont.Font(root=root, font=theme.font('body', bold=True))
        if label_font.measure(name) > 216:
            low, high = 0, len(name)
            while low < high:
                middle = (low + high + 1) // 2
                if label_font.measure(name[:middle] + '…') <= 216:
                    low = middle
                else:
                    high = middle - 1
            display_name = name[:low]
        active_group_var.set(f'Ativo · ID {state.selected_cluster}\n' +
                             (display_name + '…' if display_name != name else name))
        done = state.selected_cluster in state.completed
        background = state.selected_cluster in state.background_groups
        review_approve_btn.config(state='disabled' if done or background or state.has_pending else 'normal')
        review_reopen_btn.config(state='normal' if done else 'disabled')
        review_update_btn.config(state='normal' if state.has_pending else 'disabled')
        review_undo_btn.config(state='normal' if state.can_undo else 'disabled')
        review_reset_btn.config(state='disabled' if done else 'normal')
        group_undo_btn.config(state='normal' if state.can_undo_group else 'disabled')
        selected = selected_workspace_groups()
        one = len(selected) == 1
        for key in ('rename', 'color'):
            group_action_buttons[key].config(state='normal' if one else 'disabled')
        group_action_buttons['create'].config(state='normal')
        group_action_buttons['merge'].config(state='normal' if len(selected) >= 2 and
                                              not any(c in state.completed or c in state.background_groups for c in selected) else 'disabled')
        deletable = one and len(state.groups) > 1 and selected[0] not in state.completed and selected[0] not in state.background_groups
        group_action_buttons['delete'].config(state='normal' if deletable else 'disabled')
        activate_group_btn.config(state='normal' if selected else 'disabled')
        for tool, button in review_tool_buttons.items():
            button.config(state='disabled' if (done or background) and tool in ('positive', 'negative', 'erase') else 'normal',
                          style='Selected.Icon.TButton' if tool == review_tool_var.get() else 'Icon.TButton')
        counts = [int((item['annotations'].get(state.selected_cluster, np.zeros(0)) == value).sum())
                  for value in (1, -1) for item in state.items]
        n = len(state.items)
        review_status_var.set(
            f'{state.counts(state.selected_cluster)} px · {len(state.completed)} prontos · {len(state.background_groups)} fundo\n'
            f'Marcas: +{sum(counts[:n])} / −{sum(counts[n:])}'
            + (' · pendentes' if state.has_pending else ''))

    def clear_group_selection():
        if session['busy']:
            return
        group_tree.selection_set(())
        family_group_list.selection_clear(0, tk.END)
        on_group_selection()

    def undo_review_stroke():
        if session['busy']:
            return
        prepared = session.get('prepared')
        if prepared is None or not prepared['review'].undo():
            return
        invalidate_similarity()
        refresh_review_controls()
        refresh_group_listboxes()
        refresh_preview_content()
        status_var.set('Última pincelada desfeita. Atualize a classificação se houver edições pendentes.')

    def selected_groups_background(value):
        if session['busy']:
            return
        prepared = session.get('prepared')
        chosen = selected_workspace_groups()
        if prepared is None or not chosen:
            status_var.set('Selecione ao menos um grupo para alterar o fundo.')
            return
        try:
            prepared['review'].set_background(chosen, value)
            invalidate_similarity()
        except ValueError as exc:
            messagebox.showinfo('Fundo da análise', str(exc))
            return
        refresh_group_listboxes()
        refresh_review_controls()
        refresh_preview_content()
        status_var.set('Seleção enviada ao fundo. Clique em Atualizar classificação na Revisão.' if value else
                       'Seleção restaurada. Clique em Atualizar classificação na Revisão.')

    def reset_review_group():
        if session['busy']:
            return
        prepared = session.get('prepared')
        if prepared is None:
            return
        try:
            prepared['review'].reset_group(prepared['review'].selected_cluster)
            invalidate_similarity()
        except ValueError as exc:
            messagebox.showinfo('Resetar grupo', str(exc))
            return
        refresh_group_listboxes()
        refresh_review_controls()
        refresh_preview_content()
        status_var.set('Marcações do grupo limpas. Clique em Atualizar classificação para reiniciar seu aprendizado.')

    def select_review_image(_event=None):
        if session['busy']:
            return
        prepared = session.get('prepared')
        if prepared is None or review_image_combo.current() < 0:
            return
        prepared['review'].selected_image = review_image_combo.current()
        session['compare_cache'] = None
        rebuild_layers()
        rebuild_compare_sources()
        refresh_preview_content()

    def select_review_cluster(_event=None):
        if session['busy']:
            return
        prepared = session.get('prepared')
        chosen = [int(iid) for iid in group_tree.selection()]
        if prepared is None or not chosen:
            return
        cluster = chosen[0]
        session['workspace_selected_groups'] = chosen
        if session.get('review_after') is not None:
            root.after_cancel(session['review_after'])
            session['review_after'] = None
        prepared['review'].selected_cluster = cluster
        refresh_group_listboxes()
        refresh_review_controls()
        session['compare_cache'] = None
        rebuild_compare_sources()
        refresh_preview_content()

    def activate_selected_group(_event=None):
        if _event is not None and getattr(_event, 'state', 0) & (1 | 4):
            return select_list_item(_event)
        select_review_cluster()
        return 'break'

    def select_groups(ids, *, add=False):
        if session['busy'] or session.get('prepared') is None:
            return
        state = session['prepared']['review']
        chosen = list(session['workspace_selected_groups']) if add else []
        chosen = list(dict.fromkeys(chosen + [int(c) for c in ids if int(c) in state.groups]))
        group_tree.selection_set([str(c) for c in chosen])
        on_group_selection()

    def group_edit(method, *args, presentation=False):
        if session['busy'] or session.get('prepared') is None:
            return None
        state = session['prepared']['review']
        previous = (list(session['workspace_selected_groups']), state.selected_cluster)
        previous_snapshot = state._group_undo[-1] if state._group_undo else None
        try:
            function = getattr(state, method)
            result = (function(*args) if presentation else
                      run_operation('Atualizando os grupos...', function, *args))
        except Exception as exc:
            messagebox.showinfo('Gerenciar grupos', str(exc))
            return None
        if state._group_undo and state._group_undo[-1] is not previous_snapshot:
            session['group_selection_history'].append(previous)
        session['group_selection_history'] = session['group_selection_history'][-len(state._group_undo):] if state._group_undo else []
        if method == 'create_group':
            state.selected_cluster = result
            session['workspace_selected_groups'] = [result]
        elif method == 'merge_groups':
            session['workspace_selected_groups'] = [result]
        else:
            session['workspace_selected_groups'] = [c for c in previous[0] if c in state.groups]
        if presentation:
            refresh_similarity_windows()
        else:
            invalidate_similarity()
        session['project_dirty'] = True
        refresh_group_listboxes()
        refresh_review_controls()
        session['compare_cache'] = None
        rebuild_compare_sources()
        change_preview()
        return result

    def create_group_current():
        if session['busy'] or session.get('prepared') is None:
            return
        name = simpledialog.askstring('Criar grupo', 'Nome do novo grupo (opcional):', parent=root)
        if name is not None:
            group_edit('create_group', name)

    def rename_group_current():
        chosen = selected_workspace_groups()
        if session['busy'] or len(chosen) != 1:
            return
        name = simpledialog.askstring('Renomear grupo', 'Nome interpretativo:',
                                      initialvalue=review_group_label(chosen[0]), parent=root)
        if name is not None:
            group_edit('rename_group', chosen[0], name, presentation=True)

    def recolor_group_current():
        chosen = selected_workspace_groups()
        if session['busy'] or len(chosen) != 1:
            return
        _, color = colorchooser.askcolor(parent=root, title='Cor do grupo')
        if color is not None:
            group_edit('recolor_group', chosen[0], tuple(int(color[i:i+2], 16) for i in (1, 3, 5)), presentation=True)

    def merge_groups_current():
        chosen = selected_workspace_groups()
        if len(chosen) >= 2:
            group_edit('merge_groups', chosen)

    def delete_group_current():
        chosen = selected_workspace_groups()
        if len(chosen) == 1:
            group_edit('delete_group', chosen[0])

    def undo_group_current():
        if session['busy'] or session.get('prepared') is None:
            return
        state = session['prepared']['review']
        if not state.undo_group_operation():
            return
        selected, active = session['group_selection_history'].pop() if session['group_selection_history'] else ([], state.selected_cluster)
        session['workspace_selected_groups'] = [c for c in selected if c in state.groups]
        if active in state.groups:
            state.selected_cluster = active
        invalidate_similarity()
        session['project_dirty'] = True
        refresh_group_listboxes()
        refresh_review_controls()
        session['compare_cache'] = None
        rebuild_compare_sources()
        change_preview()

    def refine_review_cluster():
        if session['busy']:
            return
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
            trained = run_operation('Atualizando a classificação...', state.refine_pending,
                                    preferred=state.selected_cluster)
            invalidate_similarity()
            show_applied_var.set(False)
            refresh_review_controls()
            refresh_group_listboxes()
            session['compare_cache'] = None
            rebuild_layers()
            rebuild_compare_sources()
            refresh_preview_content()
            evidence = getattr(state, 'refinement_evidence', {})
            feedback = []
            if any(trained.values()):
                feedback.append('Modelo treinado; classificação atualizada.')
            if any(value.startswith(('positive', 'negative')) for value in evidence.values()):
                feedback.append('Suporte óptico local aplicado; um perfil único só permite coincidências exatas.')
            if any('conflicting' in value for value in evidence.values()):
                feedback.append('Perfis ópticos contraditórios: sem generalização nessas regiões.')
            status_var.set(' '.join(feedback) if feedback else 'Edições aplicadas; classificação atualizada.')
        except Exception as exc:
            messagebox.showerror('Revisão do grupo', str(exc))
        finally:
            refresh_review_controls()

    def approve_review_cluster():
        if session['busy']:
            return
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
        if session['busy']:
            return
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
        if session['busy']:
            return False
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
            run_operation('Salvando o projeto...', review.save_project, prepared, path)
            session['project_path'] = path
            session['project_dirty'] = False
            status_var.set(f'Projeto salvo: {path}')
            return True
        except Exception as exc:
            messagebox.showerror('Salvar projeto', str(exc))
            return False

    def open_project():
        nonlocal session, retained_legend_rows
        if session['busy']:
            return
        path = filedialog.askopenfilename(filetypes=[('Projeto RIIMA', '*.riima'), ('Todos os arquivos', '*.*')])
        if not path:
            return
        previous_status = status_var.get()
        try:
            prepared = run_operation('Carregando o projeto...', review.load_project, path)
            if session.get('prepared') is not None:
                old = session['prepared']
                old_path = session.get('project_path') or str(Path(old['output']) / 'projeto.riima')
                run_operation('Salvando a sessão anterior...', review.save_project,
                              dict(old, workspace=workspace_settings()), old_path)
                if Path(path).resolve() == Path(old_path).resolve():
                    prepared = run_operation('Carregando o projeto...', review.load_project, path)
        except Exception as exc:
            status_var.set(previous_status)
            messagebox.showerror('Abrir projeto', str(exc))
            return
        status_var.set(previous_status)
        previous = session
        controls = [(variable, variable.get()) for variable in project_view_variables]
        active = [variable.get() for variable in previous['_active_vars']]
        choices = [(combo, tuple(combo['values'])) for combo in
                   (review_image_combo, split_left_combo, split_right_combo)]
        selections = [(box, box.get(0, tk.END), box.curselection())
                      for box in (family_group_list,)]
        group_entries = [(iid, group_tree.item(iid)) for iid in group_tree.get_children()]
        group_selection = group_tree.selection()
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
            group_tree.delete(*group_tree.get_children())
            for iid, options in group_entries:
                group_tree.insert('', 'end', iid=iid, **options)
            group_tree.selection_set(group_selection)
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
        session['group_selection_history'] = []
        invalidate_similarity()
        session['project_path'] = path
        session['compare_cache'] = None
        session['viewport'] = None
        out_var.set(prepared['output'])
        clusters_var.set(prepared['config'].clusters)
        analysis_mode_var.set('Independentes (campos distintos)' if prepared['review'].mode == 'independent' else 'Registradas (mesmo campo)')
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
        restored_step = saved_workspace.get('step', 'refine' if prepared['review'].completed or prepared['review'].has_pending else 'organize')
        workflow_var.set(_STEP_ALIASES.get(restored_step, restored_step))
        if workflow_var.get() not in dict(_WORKFLOW_STEPS):
            workflow_var.set('interpretation')
        review_enabled_var.set(workflow_var.get() == 'interpretation')
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
        session['project_dirty'] = False

    def close_app():
        if session['busy']:
            status_var.set('Aguarde o processamento terminar antes de fechar.')
            return
        if session.get('prepared') is not None and not save_project_current():
            return
        root.destroy()

    def refresh_group_listboxes():
        group_tree.delete(*group_tree.get_children())
        family_group_list.delete(0, "end")
        interpretation = current_interpretation()
        if interpretation is None:
            return
        state = session['prepared']['review']
        session['workspace_selected_groups'] = [c for c in session['workspace_selected_groups'] if c in state.groups]
        counts = {}
        for item in state.items:
            ids, areas = np.unique(item['labels'][item['display_valid']], return_counts=True)
            for c, area in zip(ids, areas):
                counts[int(c)] = counts.get(int(c), 0) + int(area)
        photos = []
        for c in state.class_ids:
            photo = ImageTk.PhotoImage(Image.new('RGB', (12, 12), tuple(int(v) for v in state.palette[c])), master=root)
            photos.append(photo)
            status = ('protegido' if c in state.completed else 'fundo' if c in state.background_groups else
                      'vazio' if not counts.get(c, 0) else 'pendente' if c in state.dirty else 'em revisão')
            group_tree.insert('', 'end', iid=str(c), text=review_group_label(c), image=photo,
                              values=(c, status), tags=('active',) if c == state.selected_cluster else ())
        session['_group_swatches'] = photos
        group_tree.selection_set([str(c) for c in session['workspace_selected_groups']])
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
            name = ttk.Label(text, font=theme.font('body', bold=True),
                             wraplength=190, justify='left', anchor='w')
            name_insets = name.winfo_reqwidth()
            name.configure(text=row["label"])
            name.pack(anchor="w", fill='x')
            detail = ttk.Label(text, wraplength=190,
                               foreground=theme.TEXT_SECONDARY, justify='left', anchor='w')
            detail_insets = detail.winfo_reqwidth()
            detail.configure(text=row.get("detail", ""))
            detail.pack(anchor="w", fill='x')
            # Empty labels measure the active theme's borders and padding;
            # wraplength applies only to the text area inside those insets.
            text.bind('<Configure>', lambda event, labels=((name, name_insets), (detail, detail_insets)):
                      [label.configure(wraplength=max(1, event.width - insets)) for label, insets in labels])



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
        if session['busy']:
            session['resize_after'] = root.after(100, render_preview_canvas)
            return
        preview_canvas.delete("all")
        rgb = session.get("current_rgb")
        width = max(1, int(preview_canvas.winfo_width()))
        height = max(1, int(preview_canvas.winfo_height()))
        if rgb is None:
            preview_canvas.create_text(
                width // 2, height // 2,
                text="Selecione imagens e clique em Gerar classificação.",
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
        update_brush_outline()

    def update_brush_outline(*_args):
        position = session.get('brush_position')
        viewport = session.get('viewport')
        prepared = session.get('prepared')
        if (position is None or viewport is None or prepared is None or 'review' not in prepared
                or session['busy'] or not review_enabled_var.get() or review_tool_var.get() == 'pan'
                or session.get('compare_mode')):
            preview_canvas.delete('brush_cursor')
            return
        radius = review_radius_var.get() * viewport.zoom
        x, y = position
        box = (x - radius, y - radius, x + radius, y + radius)
        circles = preview_canvas.find_withtag('brush_cursor')
        if not circles:
            for color, width in ((theme.BG_CANVAS, 3), (theme.TEXT_ON_CANVAS, 1)):
                preview_canvas.create_oval(*box, outline=color, width=width, tags='brush_cursor')
        else:
            for circle in circles:
                preview_canvas.coords(circle, *box)
        preview_canvas.tag_raise('brush_cursor')

    def on_canvas_leave(_event=None):
        session['brush_position'] = None
        preview_canvas.delete('brush_cursor')

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
        if session['busy']:
            return
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
        if session['busy']:
            return
        session['brush_position'] = (event.x, event.y)
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
        if session['busy']:
            return
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
        invalidate_similarity()
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
        """Move native Canvas overlays without rebuilding the scientific raster."""
        session['brush_position'] = (event.x, event.y)
        update_brush_outline()
        if session['busy']:
            preview_canvas.config(cursor='watch')
            return
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
            if session['busy']:
                session['layer_opacity_after'] = root.after(100, redraw)
                return
            session['layer_opacity_after'] = None
            change_preview()
        session['layer_opacity_after'] = root.after(80, redraw)



    def update_context_controls():
        if session['busy']:
            return
        prepared = session.get('prepared')
        step = workflow_var.get()
        hints = dict(preparation='Prepare imagens e gere a classificação.',
                     interpretation='Selecione para inspecionar; ative um grupo para revisar.')
        workflow_hint_var.set(hints[step])
        for key, frame in step_frames.items():
            if key == step:
                frame.pack(fill='x', pady=(8, 0))
            else:
                frame.pack_forget()
        family_group_frame.pack_forget()
        if prepared is not None and prepared['interpretation'].families:
            legacy_masks_toggle.pack(fill='x', pady=(8, 0))
            if step == 'interpretation' and legacy_masks_var.get():
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
            available = key == 'preparation' or prepared is not None
            button.config(state='normal' if available else 'disabled',
                          style='Primary.TButton' if key == step else 'TButton')
        if prepared is not None:
            state = prepared['review']
            results_status_var.set(f'{len(state.completed)} grupos prontos e {len(state.background_groups)} de fundo.\n'
                                   + ('Há edições pendentes de atualização.' if state.has_pending else 'Classificação atualizada.'))
        refresh_export_actions()

    def change_preview():
        update_context_controls()
        update_summary_text()
        refresh_preview_content(legend=True)

    def update_summary_text():
        if session.get("prepared") is not None:
            summary_var.set(viewer.human_analysis_summary(session["prepared"]))

    def autosave_current():
        """Save the portable project; scientific products are explicit exports."""
        session['autosave_after'] = None
        if session.get('prepared') is None:
            return False
        if not prepare_output_destination():
            return False
        prepared = session['prepared']
        previous_path = session.get('project_path')
        previous_dirty = session['project_dirty']
        missing = object()
        workspace = prepared.get('workspace', missing)
        try:
            with export_batch(publish=lambda write: run_operation('Publicando o projeto...', write)):
                if not save_project_current():
                    raise OSError('Não foi possível salvar o projeto.')
            status_var.set('Projeto salvo. Exporte os produtos desejados pelo menu superior.')
            return True
        except Exception as exc:
            session['project_path'] = previous_path
            session['project_dirty'] = previous_dirty
            if workspace is missing:
                prepared.pop('workspace', None)
            else:
                prepared['workspace'] = workspace
            status_var.set('Salvamento do projeto pendente.')
            messagebox.showerror('Salvar projeto', str(exc))
            return False

    def export_scientific_current():
        """Explicit full scientific tree, published with its portable project."""
        if session['busy']:
            return False
        session["autosave_after"] = None
        if session.get("prepared") is None:
            return
        if not prepare_output_destination():
            return False
        prepared = session['prepared']
        previous_path = session.get('project_path')
        previous_dirty = session['project_dirty']
        missing = object()
        previous_workspace = prepared.get('workspace', missing)
        try:
            with export_batch(publish=lambda write: run_operation('Publicando os resultados...', write)):
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
                        run_operation('Salvando os resultados iniciais...', core.save_session_outputs,
                                      dict(prepared, optical=item['optical'], interpretation=interpretation,
                                           output=str(Path(prepared['output']) / 'initial' / f'image_{i:03d}')))
                else:
                    run_operation('Salvando os resultados iniciais...', core.save_session_outputs, prepared)
                if 'review' in prepared:
                    run_operation('Salvando os resultados da revisão...', review.save_review_outputs, prepared)
                run_operation('Calculando e exportando as métricas...', metrics.export_metrics, prepared)
            status_var.set("Exportação científica completa concluída.")
            return True
        except Exception as exc:
            session['project_path'] = previous_path
            session['project_dirty'] = previous_dirty
            if previous_workspace is missing:
                prepared.pop('workspace', None)
            else:
                prepared['workspace'] = previous_workspace
            status_var.set("A análise está pronta, mas a exportação científica falhou.")
            messagebox.showerror("Salvar resultados", str(exc))
            return False

    def schedule_autosave():
        """Keep organization changes in memory until explicit save or normal close."""
        if session.get("autosave_after") is not None:
            root.after_cancel(session["autosave_after"])
        session['autosave_after'] = None
        save_interpretation_metadata()

    def save_interpretation_metadata():
        """Legacy callback: record unsaved interpretation without technical sidecars."""
        session['autosave_after'] = None
        prepared = session.get('prepared')
        if prepared is None:
            return
        session['project_dirty'] = True
        status_var.set('Alterações no projeto; use Salvar projeto para guardar o progresso.')

    def refresh_interpretation():
        if session['busy']:
            return
        refresh_similarity_windows()
        refresh_review_controls()
        refresh_group_listboxes()
        update_summary_text()
        session["compare_cache"] = None
        rebuild_layers()
        rebuild_compare_sources()
        change_preview()
        schedule_autosave()

    def apply_interpretation(method, *args):
        if session['busy']:
            return
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

    def legacy_selected_indices():
        prepared = session.get('prepared')
        if prepared is None:
            return []
        members = [prepared['review'].groups[int(c)] for c in group_tree.selection()]
        return [i for i, group in enumerate(prepared['interpretation'].groups) if group in members]

    def merge_optical_selected():
        selected = legacy_selected_indices()
        if len(selected) < 2:
            messagebox.showinfo("Unir clusters", "Selecione pelo menos dois clusters.")
            return
        apply_interpretation("merge_clusters", selected)

    def undo_optical_groups():
        apply_interpretation("undo")

    def reset_optical_groups():
        apply_interpretation("reset_clusters")

    def create_family_selected():
        selected = legacy_selected_indices()
        if not selected:
            messagebox.showinfo("Criar família", "Selecione um ou mais clusters acima.")
            return
        name = simpledialog.askstring("Criar família", "Nome da família (opcional):", parent=root)
        if name is not None:
            apply_interpretation("create_family", selected, name.strip())

    def select_legacy_mask():
        if session['busy']:
            return
        prepared = session.get('prepared')
        selected = family_group_list.curselection()
        if prepared is None or not selected:
            return
        members = set().union(*(set(prepared['interpretation'].families[i]['clusters']) for i in selected))
        state = prepared['review']
        chosen = [c for c in state.class_ids if state.groups[c] and state.groups[c] <= members]
        if not chosen:
            return
        session['workspace_selected_groups'] = chosen
        review_display_var.set('Máscara selecionada')
        isolate_var.set(True)
        refresh_review_controls()
        refresh_group_listboxes()
        session['compare_cache'] = None
        rebuild_compare_sources()
        change_preview()

    def change_family_membership(method):
        families = list(family_group_list.curselection())
        clusters = legacy_selected_indices()
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
        selected = legacy_selected_indices() if listbox is group_tree else list(listbox.curselection())
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
        selected = legacy_selected_indices() if listbox is group_tree else list(listbox.curselection())
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
            if session['busy']:
                win.after(100, run_evaluation)
                return
            try:
                result = run_operation('Avaliando os valores de K...', core.evaluate_cluster_counts,
                                       sample_X, k_values, seed=13)
            except Exception as exc:
                if win.winfo_exists():
                    status_lbl.config(text=str(exc))
                return
            status_var.set('Diagnóstico de K concluído.')
            if not win.winfo_exists():
                return
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
        if session['busy']:
            return
        images = session.get('images', [])
        if not images:
            messagebox.showwarning('Entrada incompleta', 'Selecione uma ou mais imagens.')
            return
        flags = [v.get() for v in session.get('_active_vars', [])]
        independent = analysis_mode_var.get().startswith('Independentes')
        try:
            if independent:
                selected_paths = [item['path'] for i, item in enumerate(images) if not flags or flags[i]]
                if not selected_paths:
                    raise ValueError('Selecione ao menos uma imagem ativa.')
                ref_idx = 0
            else:
                selected_paths, ref_idx = core.normalize_image_selection(
                    [item['path'] for item in images], flags, int(reference_var.get()))
        except (ValueError, tk.TclError) as exc:
            messagebox.showwarning('Seleção de imagens', str(exc))
            return
        try:
            optical_cfg = core.Config(clusters=int(clusters_var.get()))
        except (ValueError, tk.TclError) as exc:
            messagebox.showerror('Configuração da análise', str(exc))
            return
        if not out_var.get():
            out_var.set(str(Path(selected_paths[0]).parent / 'riima_output'))
        if session.get('prepared') is not None:
            if not messagebox.askyesno('Substituir análise?',
                    'Uma nova classificação substituirá os grupos e a revisão atuais. '
                    'O projeto anterior será salvo e preservado em seu arquivo. Continuar?', parent=root):
                return
            if not save_project_current():
                return
        previous = dict(session)
        previous_variables = [(variable, variable.get()) for variable in project_view_variables]
        start_time = time.perf_counter()
        mask_mode = _MASK_MODE_LABELS.get(mask_mode_var.get(), 'auto')
        output = out_var.get()
        analyze_btn.config(state='disabled')
        def prepare_candidate():
            if independent:
                result = review.prepare_independent_analysis(selected_paths, output, optical_cfg,
                    mask_mode=mask_mode, stage_callback=set_status)
            else:
                result = core.prepare_analysis(selected_paths, output, reference_index=ref_idx,
                    optical_cfg=optical_cfg, mask_mode=mask_mode, stage_callback=set_status, publish_outputs=False)
            if 'review' not in result:
                result['review'] = review.ClusterReview.from_prepared(result)
            return result
        try:
            prepared = run_operation('Preparando a nova análise...', prepare_candidate)
            prepared['workspace'] = dict(workspace_settings(), selected_groups=[prepared['review'].selected_cluster],
                                         display='Sobreposição', isolate=False, contours=False,
                                         show_applied=False, view_type='layers')
            target = Path(prepared['output']) / 'projeto.riima'
            suffix = 2
            while target.exists():
                target = Path(prepared['output']) / f'projeto_{suffix}.riima'
                suffix += 1
            # Complete and publish the candidate archive before replacing the active
            # scientific session. A failed preparation/write leaves the old result intact.
            with export_batch(publish=lambda write: run_operation('Publicando o novo projeto...', write)):
                run_operation('Salvando a nova análise...', review.save_project, prepared, str(target))
            show_loaded_project(prepared, str(target))
            session['images'] = previous['images']
            rebuild_image_rows()
            for variable, enabled in zip(session['_active_vars'], flags):
                variable.set(enabled)
            reference_var.set(int(next(value for variable, value in previous_variables if variable is reference_var)))
            status_map = {}
            for entry in prepared['optical'].get('registered_images', []):
                reg = entry.get('registration')
                status_map[entry['name']] = ('imagem incluída' if reg is None else 'referência' if reg.get('is_reference') else
                    'Confira o alinhamento: coordenadas preservadas; registro não verificado.' if reg.get('registration_warning') else
                    core.registration_quality_label(reg['score_before'], reg['score_after']))
            update_row_statuses(status_map)
            session['analysis_seconds'] = time.perf_counter() - start_time
            note = ''
            raw = prepared['optical']
            if raw.get('overlap_status') == 'warn':
                note += ' Atenção: sobreposição comum entre as imagens está baixa.'
            overlap, valid = raw.get('overlap'), raw.get('valid')
            if mask_mode == 'auto' and overlap is not None and valid is not None and overlap.mean() > 0:
                if float(valid.mean()) / float(overlap.mean()) < _MASK_COVERAGE_WARN_RATIO:
                    note += ' A máscara automática excluiu uma fração grande da área registrada; confira a máscara e compare com Somente sobreposição comum.'
            status_var.set(f"Análise concluída em {session['analysis_seconds']:.1f} s. Projeto salvo: {target.name}.{note}")
        except Exception as exc:
            session.clear()
            session.update(previous)
            for variable, value in previous_variables:
                variable.set(value)
            rebuild_image_rows()
            for variable, enabled in zip(session['_active_vars'], flags):
                variable.set(enabled)
            if session.get('prepared') is not None:
                setup_review()
                refresh_group_listboxes()
                rebuild_layers()
                rebuild_compare_sources()
            change_preview()
            status_var.set('Não foi possível substituir a análise. O estado anterior foi preservado.')
            messagebox.showerror('Erro na análise', str(exc))
        finally:
            analyze_btn.config(state='normal')
            refresh_export_actions()


    def save_png_export(rgb, path, title):
        """Complete a PNG before replacing an existing user-selected export."""
        import os
        import tempfile
        temporary = None
        def write_png():
            nonlocal temporary
            with export_batch():
                destination = staged_file(path)
                with tempfile.NamedTemporaryFile(prefix='.riima-png-', suffix='.png',
                                                 dir=destination.parent, delete=False) as file:
                    temporary = Path(file.name)
                Image.fromarray(rgb).save(temporary, format='PNG', compress_level=3, optimize=False)
                os.replace(temporary, destination)
        try:
            run_operation('Exportando o PNG...', write_png)
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
            directory = run_operation('Calculando e exportando as métricas...', metrics.export_metrics, prepared)
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
            exported = run_operation('Gerando o relatório...', riima_report.export_report,
                                     prepared, path, stage_callback=set_status)
            status_var.set(f'Relatório descritivo exportado: {exported}')
        except Exception as exc:
            messagebox.showerror('Exportar relatório', str(exc))
            status_var.set(f'Não foi possível exportar o relatório: {exc}')

    def refresh_export_actions():
        prepared = session.get('prepared')
        available = prepared is not None and not session['busy']
        applied = available and not prepared['review'].has_pending
        save_project_btn.config(state='normal' if available else 'disabled')
        export_menu_btn.config(state='normal' if available else 'disabled')
        for index, enabled in enumerate((applied, available and session.get('current_rgb') is not None,
                                          available and bool(selected_workspace_groups()), applied, applied,
                                          available, available)):
            export_menu.entryconfigure(index, state='normal' if enabled else 'disabled')
        similarity_btn.config(state='normal' if applied and bool(selected_workspace_groups()) else 'disabled')

    def group_similarity_rows(data, selected):
        """Resolve presentation from current IDs, keeping cached calculations unchanged."""
        prepared = session['prepared']
        state = prepared['review']
        active_ids = [data['groups'][i]['id'] for i in data['active']]
        if selected not in active_ids:
            return []
        row = active_ids.index(selected)
        result = [dict(id=c, name=review_group_label(c), color=tuple(int(v) for v in state.palette[c]),
                       similarity=float(data['similarity'][row, i]))
                  for i, c in enumerate(active_ids) if c != selected and c in state.groups]
        return sorted(result, key=lambda item: (-item['similarity'], item['id']))

    def open_group_similarity():
        if session['busy']:
            return
        prepared = session.get('prepared')
        chosen = selected_workspace_groups()
        if prepared is None or not chosen:
            messagebox.showinfo('Similaridade', 'Selecione um grupo para comparar com os demais.')
            return
        if prepared['review'].has_pending:
            messagebox.showinfo('Similaridade', 'Atualize a classificação antes de comparar os grupos.')
            return
        selected = chosen[0]  # Capture before the worker dispatches Tk events.
        revision = session['science_revision']
        window = tk.Toplevel(root)
        similarity_windows.add(window)
        def forget_window(event):
            if event.widget is window:
                similarity_windows.discard(window)
        window.bind('<Destroy>', forget_window, add='+')
        window.title('Similaridade entre grupos')
        window.transient(root)
        window.geometry('480x360')
        window.minsize(380, 280)
        panel = ttk.Frame(window, padding=12)
        panel.pack(fill='both', expand=True)
        group_heading = ttk.Label(panel, text=review_group_label(selected), style='Section.TLabel', wraplength=440)
        group_heading.pack(anchor='w')
        explanation = ttk.Label(panel, text='Afinidade relativa nesta análise; não indica identidade mineralógica.',
                                 wraplength=440, style='Muted.TLabel')
        explanation.pack(anchor='w', pady=(4, 8))
        tree = ttk.Treeview(panel, columns=('affinity',), show='tree headings', height=7)
        tree.heading('#0', text='Grupo comparado')
        tree.heading('affinity', text='Afinidade')
        tree.column('#0', width=310, minwidth=180)
        tree.column('affinity', width=85, minwidth=75, anchor='e', stretch=False)
        tree.pack(fill='both', expand=True)
        note = ttk.Label(panel, text='Calculando perfis atuais...', wraplength=440, style='Muted.TLabel')
        note.pack(anchor='w', pady=(6, 0))
        window.bind('<Escape>', lambda _e: window.destroy())
        try:
            cache = session.get('similarity_cache')
            if cache is not None and cache['prepared'] is prepared and cache['revision'] == revision:
                data = cache['data']
            else:
                from . import report_data
                data = run_operation('Comparando os grupos atuais...', report_data.collect_report_data, prepared)
                if session.get('prepared') is not prepared or session['science_revision'] != revision:
                    if window.winfo_exists():
                        window.destroy()
                    return
                session['similarity_cache'] = dict(prepared=prepared, revision=revision, data=data)
            if not window.winfo_exists():
                return
            by_id = {}
            def sync_selection():
                chosen = selected_workspace_groups()
                for iid in tree.get_children():
                    tree.item(iid, tags=('shared',) if int(iid) in chosen else ())
            tree.tag_configure('shared', background=theme.ACCENT_SOFT)
            def redraw():
                group_heading.config(text=review_group_label(selected))
                rows = group_similarity_rows(data, selected)
                session['similarity_rows'] = rows
                by_id.clear()
                by_id.update({str(row['id']): row for row in rows})
                tree.delete(*tree.get_children())
                photos = []
                for row in rows:
                    swatch = ImageTk.PhotoImage(Image.new('RGB', (12, 12), row['color']), master=root)
                    photos.append(swatch)
                    tree.insert('', 'end', iid=str(row['id']), text=row['name'], image=swatch,
                                values=(f"{row['similarity']:.3f}".replace('.', ','),))
                window._swatches = photos
                sync_selection()
                note.config(text='Perfil: geometria, atributos ópticos e direção axial. Valores de 0 a 1.' if rows else
                            'Sem dados para comparar este grupo; grupos vazios e de fundo ficam fora do cálculo.')
            window._refresh, window._sync_selection = redraw, sync_selection
            redraw()
            def show_selected_name(_event=None):
                selected_rows = tree.selection()
                if selected_rows:
                    row = by_id[selected_rows[0]]
                    affinity = f"{row['similarity']:.3f}".replace('.', ',')
                    note.config(text=f"{row['name']}\nAfinidade: {affinity}")
                    select_groups([row['id']], add=True)
            tree.bind('<<TreeviewSelect>>', show_selected_name)
            def wrap_labels(event):
                if event.widget is window:
                    width = max(180, event.width - 28)
                    for label in (group_heading, explanation, note):
                        label.config(wraplength=width)
            window.bind('<Configure>', wrap_labels, add='+')
            tree.focus_set()
            return window
        except Exception as exc:
            if window.winfo_exists():
                note.config(text=f'Não foi possível comparar: {exc}')
            status_var.set(f'Não foi possível calcular a similaridade: {exc}')
            return window if window.winfo_exists() else None

    def open_mask_comparison():
        """Inspect the stored analysis mask against its actual geometric support."""
        prepared = session.get('prepared')
        if prepared is None or session['busy']:
            return
        state = prepared['review']
        independent = state.mode == 'independent'
        item = state.items[state.selected_image if independent else 0]
        raw = item['optical'] if independent else prepared['optical']
        window = tk.Toplevel(root)
        window.title('Conferir máscara de análise')
        window.transient(root)
        panel = ttk.Frame(window, padding=12)
        panel.pack(fill='both', expand=True)
        window.bind('<Escape>', lambda _e: window.destroy())
        overlap = raw.get('overlap')
        if overlap is None and not independent:
            ttk.Label(panel, text='Comparação indisponível: este projeto histórico não preservou a sobreposição '
                      'geométrica. A máscara atual foi mantida sem supor outra área válida.',
                      wraplength=470).pack()
            return window
        valid = np.asarray(raw['valid'], dtype=bool)
        if overlap is None:
            overlap = np.ones(valid.shape, bool)
        rgb = item['rgb'] if independent else raw['images'][state.selected_image]
        ttk.Label(panel, text='Máscara usada × domínio geométrico', style='Section.TLabel').pack(anchor='w')
        ttk.Label(panel, text=('Campo independente: o domínio geométrico é a imagem inteira.' if independent else
                  'A sobreposição limita o domínio válido do registro; a máscara automática é uma heurística de fundo.'),
                  wraplength=640, style='Muted.TLabel').pack(anchor='w', pady=(4, 8))
        previews = ttk.Frame(panel)
        previews.pack(fill='both', expand=True)
        window._mask_photos = []
        for column, (title, mask) in enumerate((('Máscara usada na análise', valid),
                                               ('Somente domínio geométrico', np.asarray(overlap, bool)))):
            frame = ttk.Frame(previews)
            frame.grid(row=0, column=column, padx=(0, 10) if column == 0 else (0, 0), sticky='n')
            pixels = int(mask.sum())
            ttk.Label(frame, text=f'{title}\n{pixels} pixels ({100 * pixels / max(1, mask.size):.1f}%)',
                      justify='left').pack(anchor='w')
            overlay = np.array(rgb, copy=True)
            overlay[~mask] //= 5
            image = Image.fromarray(overlay)
            image.thumbnail((300, 280), Image.Resampling.NEAREST)
            photo = ImageTk.PhotoImage(image, master=root)
            window._mask_photos.append(photo)
            ttk.Label(frame, image=photo).pack(pady=(6, 0))
        ttk.Label(panel, text='Áreas fora de cada máscara aparecem escurecidas. A comparação usa a máscara da análise '
                  'inicial; exclusões manuais da revisão são preservadas separadamente.', wraplength=640,
                  style='Muted.TLabel').pack(anchor='w', pady=(8, 0))
        def choose_geometric():
            mask_mode_var.set('Somente sobreposição comum')
            status_var.set('Máscara selecionada para a próxima classificação; resultado atual preservado.')
            window.destroy()
        action_button(panel, 'Usar somente sobreposição na próxima análise', choose_geometric,
                      description='Apenas seleciona a opção. Clique em Gerar classificação quando desejar recalcular.').pack(anchor='w', pady=(8, 0))
        return window

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
    open_project_btn = action_button(project_actions, 'Abrir projeto...', open_project, icon='open',
                                     description='Retoma imagens, marcações e classificação de um arquivo .riima.')
    open_project_btn.pack(side='left', padx=3)
    save_project_btn = action_button(project_actions, 'Salvar projeto', save_project_current, icon='save',
                                     description='Guarda o progresso e as marcações pendentes no projeto portátil.')
    save_project_btn.pack(side='left', padx=3)
    export_menu = tk.Menu(root, tearoff=False, font=theme.FONT_BODY, background=theme.BG_PANEL,
                          foreground=theme.TEXT_PRIMARY, postcommand=refresh_export_actions)
    for label, command in [('Relatório descritivo (HTML)', export_descriptive_report),
                           ('Visualização atual (PNG)', export_current),
                           ('Máscara selecionada (PNG)', export_selected_mask),
                           ('Exportação científica completa', export_scientific_current),
                           ('Métricas (CSV)', export_light_metrics),
                           ('Salvar projeto como...', lambda: save_project_current(choose=True)),
                           ('Abrir pasta de resultados', open_output_folder)]:
        export_menu.add_command(label=label, command=command)
    export_menu_btn = ttk.Menubutton(project_actions, text='Exportar', image=desktop_icon('export'),
                                    compound='left', menu=export_menu, style='Toolbar.TMenubutton', takefocus=True)
    attach_tooltip(export_menu_btn, 'Exportar — Escolha relatório, imagem, máscara, métricas ou a árvore científica completa.')
    export_menu_btn.pack(side='left', padx=3)
    root.bind('<Control-s>', lambda _e: save_project_current() if not session['busy'] else None)
    root.bind('<Control-o>', lambda _e: open_project() if not session['busy'] else None)
    root.bind('<Control-Shift-S>', lambda _e: save_project_current(choose=True) if not session['busy'] else None)
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
    sidebar_container = ttk.Frame(main, width=272)
    sidebar_container.grid(row=0, column=0, sticky="nsw")
    sidebar_container.grid_propagate(False)
    sidebar_container.pack_propagate(False)
    sidebar_container.configure(width=272)

    sidebar_canvas = tk.Canvas(sidebar_container, highlightthickness=0, bd=0, bg=theme.BG_APP)
    sidebar_scroll = ttk.Scrollbar(sidebar_container, orient="vertical", command=sidebar_canvas.yview)
    sidebar_canvas.configure(yscrollcommand=sidebar_scroll.set)
    sidebar_scroll.pack(side="right", fill="y")
    sidebar_canvas.pack(side="left", fill="both", expand=True)

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


    ttk.Label(sidebar, textvariable=workflow_hint_var, wraplength=216,
              foreground=theme.TEXT_SECONDARY).pack(anchor='w', pady=(0, 8))
    step_frames = {key: ttk.Frame(sidebar) for key, _ in _WORKFLOW_STEPS}
    images_step = classify_step = step_frames['preparation']
    organize_step = refine_step = step_frames['interpretation']
    results_status_var = tk.StringVar()
    interpretation_slot = organize_step  # Compatibility for the existing callbacks.
    ttk.Label(images_step, text="Imagens", style="Section.TLabel").pack(anchor="w", pady=(0, 6))
    ttk.Button(images_step, text="Selecionar imagens...", command=choose_images).pack(fill="x", pady=(0, 4))
    images_list_frame = ttk.Frame(images_step)
    images_list_frame.pack(fill="x", pady=(0, 2))
    ttk.Label(images_list_frame, text="Nenhuma imagem selecionada.", foreground=theme.TEXT_SECONDARY).pack(anchor="w")
    context_help(images_step, 'Selecione imagens do mesmo campo. Todas as imagens ativas contribuem para a análise; '
                 'a referência serve ao alinhamento. Uma imagem isolada usa o mesmo fluxo.').pack(anchor='w', pady=(2, 6))

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
        wraplength=216, foreground=theme.TEXT_SECONDARY,
    ).pack(anchor="w", pady=(0, 6))
    clusters_row = ttk.Frame(classify_step); clusters_row.pack(fill="x", pady=(0, 4))
    ttk.Label(clusters_row, text="Clusters ópticos").pack(side="left")
    ttk.Spinbox(clusters_row, from_=2, to=12, textvariable=clusters_var, width=6).pack(side="right")
    analysis_mode_combo = ttk.Combobox(classify_step, textvariable=analysis_mode_var, state='readonly',
                                      values=['Registradas (mesmo campo)', 'Independentes (campos distintos)'])
    analysis_mode_combo.pack(fill='x', pady=(0, 5))
    attach_tooltip(analysis_mode_combo, 'Modo de análise — Registradas alinha imagens do mesmo campo. '
                   'Independentes compartilha a classificação óptica entre campos sem alinhá-los.')
    mask_row = ttk.Frame(classify_step); mask_row.pack(fill="x", pady=(0, 4))
    ttk.Label(mask_row, text="Máscara de análise").pack(side="left")
    mask_combo = ttk.Combobox(
        mask_row, textvariable=mask_mode_var, state="readonly", width=15,
        values=list(_MASK_MODE_LABELS.keys()),
    )
    mask_combo.pack(side="right")
    attach_tooltip(mask_combo, 'Máscara de análise — Automática busca remover fundo e anotações. '
                   'Compare com Somente sobreposição comum se alguma área válida for excluída.')

    analyze_btn = ttk.Button(classify_step, text="Gerar classificação", style="Primary.TButton", command=analyze)
    analyze_btn.pack(fill="x", pady=(2, 6))
    ttk.Label(
        classify_step,
        text="O projeto é salvo automaticamente; exportações ficam no menu superior.",
        wraplength=216, foreground=theme.TEXT_SECONDARY,
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
    attach_tooltip(diagnostics_frame.winfo_children()[0],
                   'Diagnóstico de K — Compara separação estatística; a escolha final cabe ao especialista.')
    action_button(diagnostics_frame, 'Conferir máscara', open_mask_comparison,
                  description='Compare a máscara da análise com o domínio geométrico sem recalcular ou exportar.').pack(fill='x', pady=(4, 0))

    ttk.Separator(classify_step, orient="horizontal").pack(fill="x", pady=12)
    optical_group_frame = ttk.LabelFrame(organize_step, text="Grupos", padding=theme.SPACE_SM)
    group_list_frame = ttk.Frame(optical_group_frame)
    group_list_frame.pack(fill='x')
    group_tree = ttk.Treeview(group_list_frame, columns=('id', 'status'), show='tree headings',
                              height=5, selectmode='extended')
    optical_group_list = group_tree  # The historical internal name refers to the same real widget.
    group_tree.heading('#0', text='Grupo'); group_tree.heading('id', text='ID'); group_tree.heading('status', text='Estado')
    group_tree.column('#0', width=100, minwidth=55, stretch=True)
    group_tree.column('id', width=28, minwidth=28, stretch=False, anchor='e')
    group_tree.column('status', width=74, minwidth=55, stretch=False)
    group_tree.tag_configure('active', font=theme.font('body', bold=True))
    group_tree.pack(side='left', fill='both', expand=True)
    group_scrollbar = ttk.Scrollbar(group_list_frame, orient='vertical', command=group_tree.yview)
    group_scrollbar.pack(side='right', fill='y')
    group_tree.config(yscrollcommand=group_scrollbar.set)
    def scroll_groups(event):
        direction = -int(event.delta / 120) if getattr(event, 'delta', 0) else (-1 if event.num == 4 else 1)
        group_tree.yview_scroll(direction, 'units')
        return 'break'
    group_tree.bind('<MouseWheel>', scroll_groups)
    group_tree.bind('<Button-4>', scroll_groups)
    group_tree.bind('<Button-5>', scroll_groups)
    group_tree.bind('<<TreeviewSelect>>', on_group_selection)
    group_tree.bind('<Button-1>', select_list_item)
    group_tree.bind('<KeyPress>', lambda _e: 'break' if session['busy'] else None)
    group_tree.bind('<B1-Motion>', lambda _e: 'break' if session['busy'] else None)
    group_tree.bind('<Double-Button-1>', activate_selected_group)
    group_tree.bind('<Return>', activate_selected_group)
    attach_tooltip(group_tree, 'Grupos — Clique seleciona; Shift/Ctrl + clique acrescenta ou remove. '
                   'Enter ou duplo clique ativa o grupo para o pincel. A seleção para inspeção é independente do grupo ativo.')
    manager_actions = ttk.Frame(optical_group_frame)
    manager_actions.pack(fill='x', pady=(4, 0))
    group_action_buttons = {}
    for text, icon, command in [('Criar grupo', 'create', create_group_current),
                                ('Renomear grupo', 'rename', rename_group_current),
                                ('Alterar cor', 'color', recolor_group_current),
                                ('Unir grupos', 'merge', merge_groups_current),
                                ('Excluir grupo', 'delete', delete_group_current)]:
        button = action_button(manager_actions, text, command, icon=icon, compound='image')
        button.config(compound='image', style='Icon.TButton')
        button.pack(side='left', padx=(0, 3))
        group_action_buttons[icon] = button
    group_undo_btn = action_button(manager_actions, 'Desfazer grupos', undo_group_current, icon='undo')
    group_undo_btn.config(compound='image', style='Icon.TButton')
    group_undo_btn.pack(side='left')
    selection_actions = ttk.Frame(optical_group_frame)
    selection_actions.pack(fill='x', pady=(4, 0))
    clear_selection_btn = action_button(selection_actions, 'Limpar seleção', clear_group_selection, icon='clear',
                                        description='Limpa a seleção sem alterar a classificação.')
    clear_selection_btn.config(compound='image', style='Icon.TButton'); clear_selection_btn.pack(side='left')
    similarity_btn = action_button(selection_actions, 'Similaridade', open_group_similarity, icon='similarity',
                                   description='Compara o primeiro grupo selecionado com os demais, usando o estado aplicado.')
    similarity_btn.pack(side='left', padx=(4, 0))
    similarity_btn.config(compound='image', style='Icon.TButton')
    activate_group_btn = action_button(selection_actions, 'Ativar grupo', activate_selected_group, icon='activate',
                                       description='Escolhe o primeiro grupo selecionado como destino das marcações.')
    activate_group_btn.config(compound='image', style='Icon.TButton'); activate_group_btn.pack(side='left', padx=4)
    background_actions = selection_actions
    for text, icon, value in [('Enviar seleção ao fundo', 'background', True), ('Restaurar seleção do fundo', 'restore', False)]:
        button = action_button(background_actions, text, lambda chosen=value: selected_groups_background(chosen), icon=icon)
        button.config(compound='image', style='Icon.TButton'); button.pack(side='left', padx=(0, 4))

    optical_group_frame.pack(fill="x")
    review_frame = ttk.LabelFrame(refine_step, text='Revisão dos grupos', padding=theme.SPACE_SM)
    active_group_label = ttk.Label(review_frame, textvariable=active_group_var, wraplength=216,
                                    font=theme.font('body', bold=True))
    active_group_label.pack(fill='x', pady=(0, 4))
    attach_tooltip(active_group_label, lambda: 'Grupo ativo: ' + review_group_label(session['prepared']['review'].selected_cluster))
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
        if session['busy']:
            return
        review_tool_var.set(tool)
        activate_review_tool()
    tool_descriptions = {
        'positive': 'Exemplos positivos de aparência do grupo; aplique em Atualizar classificação.',
        'negative': 'Exemplos de aparência que o grupo deve rejeitar; preserva os demais grupos.',
        'pan': 'Arraste a imagem sem alterar exemplos ou exclusões.',
        'erase': 'Apagar marcas — Limpa exemplos positivos e negativos do grupo; preserva o fundo.',
        'background': 'Exclui pixels da análise na próxima atualização; respeita grupos prontos.',
        'restore': 'Restaurar área — Reverte exclusões de fundo sem apagar exemplos.',
    }
    for index, (text, value) in enumerate([('Manter', 'positive'), ('Remover', 'negative'), ('Apagar marcas', 'erase'),
                                          ('Fundo', 'background'), ('Restaurar área', 'restore'), ('Navegar', 'pan')]):
        button = action_button(tools_row, text, lambda tool=value: choose_review_tool(tool),
                                icon=value, description=tool_descriptions[value])
        button.config(compound='image', style='Icon.TButton')
        button.grid(row=0, column=index, sticky='ew', pady=2, padx=(0, 2))
        review_tool_buttons[value] = button
    radius_row = ttk.Frame(review_frame)
    radius_row.pack(fill='x')
    ttk.Label(radius_row, text='Raio (px da imagem)').pack(side='left')
    ttk.Label(radius_row, textvariable=review_radius_var, width=3, anchor='e').pack(side='right')

    def set_review_radius(value):
        radius = min(150, max(1, round(float(value))))
        if review_radius_var.get() != radius:
            review_radius_var.set(radius)

    review_radius_scale = ttk.Scale(review_frame, from_=1, to=150, command=set_review_radius)
    review_radius_scale.pack(fill='x')

    def sync_review_radius(*_args):
        radius = min(150, max(1, review_radius_var.get()))
        review_radius_var.set(radius)
        review_radius_scale.set(radius)
        update_brush_outline()

    review_radius_var.trace_add('write', sync_review_radius)
    review_tool_var.trace_add('write', update_brush_outline)
    review_enabled_var.trace_add('write', update_brush_outline)

    def radius_key(event):
        value = {'Home': 1, 'End': 150}.get(event.keysym)
        if value is None:
            value = review_radius_var.get() + (-1 if event.keysym in ('Left', 'Down') else 1)
        set_review_radius(value)
        return 'break'

    for key in ('Left', 'Right', 'Up', 'Down', 'Home', 'End'):
        review_radius_scale.bind('<' + key + '>', radius_key)
    review_update_btn = ttk.Button(review_frame, text='Atualizar classificação', command=refine_review_cluster)
    review_update_btn.pack(fill='x', pady=(3, 0))
    correction_actions = ttk.Frame(review_frame)
    correction_actions.pack(fill='x', pady=(3, 0))
    review_undo_btn = action_button(correction_actions, 'Desfazer', undo_review_stroke, icon='undo',
                                     description='Desfaz a última pincelada, incluindo todos os seus segmentos.')
    review_undo_btn.pack(side='left')
    review_undo_btn.config(compound='image', style='Icon.TButton')
    review_reset_btn = action_button(correction_actions, 'Resetar grupo', reset_review_group,
                                      description='Limpa exemplos e reinicia o aprendizado na próxima atualização.')
    review_reset_btn.pack(side='left', padx=4)
    review_reset_btn.config(image=desktop_icon('clear'), compound='image', style='Icon.TButton')
    ttk.Checkbutton(review_frame, text='Mostrar marcações', variable=show_applied_var,
                    command=change_preview).pack(anchor='w', pady=(4, 0))
    approval_row = ttk.Frame(review_frame)
    approval_row.pack(fill='x', pady=4)
    review_approve_btn = action_button(approval_row, 'Aprovar grupo', approve_review_cluster, icon='activate',
                                       description='Protege a máscara aplicada do grupo ativo.')
    review_approve_btn.config(compound='image', style='Icon.TButton')
    review_approve_btn.pack(side='left')
    review_reopen_btn = action_button(approval_row, 'Reabrir grupo', reopen_review_cluster, icon='undo')
    review_reopen_btn.config(compound='image', style='Icon.TButton')
    review_reopen_btn.pack(side='left', padx=5)
    ttk.Label(review_frame, textvariable=review_status_var, wraplength=216, style='Muted.TLabel').pack(anchor='w')
    review_help_btn = context_help(approval_row, 'As marcações ficam pendentes até Atualizar classificação. Grupo pronto protege toda a seleção. '
                 'Salvar projeto guarda também exemplos pendentes. Ctrl+S salva; Ctrl+O abre um projeto.')
    review_help_btn.config(compound='image', style='Icon.TButton')
    review_help_btn.pack(side='right')

    legacy_masks_toggle = ttk.Checkbutton(organize_step, text='Seleções de projetos antigos',
                                         variable=legacy_masks_var, command=change_preview)
    family_group_frame = ttk.LabelFrame(organize_step, text="Associações salvas", padding=theme.SPACE_SM)
    ttk.Label(family_group_frame, text="Use as associações existentes para selecionar máscaras dos grupos atuais.", wraplength=216, foreground=theme.TEXT_SECONDARY).pack(anchor="w", pady=(0, 5))
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

    # ---------- central preview + fixed legend viewport ----------
    workspace = ttk.Frame(main)
    workspace.grid(row=0, column=1, sticky="nsew")
    workspace.columnconfigure(0, weight=1)
    workspace.rowconfigure(1, weight=1)

    titlebar = ttk.Frame(workspace)
    titlebar.grid(row=0, column=0, sticky="ew", padx=(4, 0), pady=(2, 5))
    titlebar.columnconfigure(0, weight=1)
    preview_title_label = ttk.Label(titlebar, textvariable=preview_title_var,
                                    font=theme.font('viewer', bold=True), wraplength=320)
    preview_title_label.grid(row=0, column=0, sticky='ew', padx=(0, 8))

    zoom_toolbar = ttk.Frame(titlebar)
    zoom_toolbar.grid(row=0, column=1, sticky='e')
    action_button(zoom_toolbar, 'Diminuir zoom', lambda: zoom_button(1 / 1.25), icon='zoom_out', compound='image').pack(side='left', padx=1)
    action_button(zoom_toolbar, 'Aumentar zoom', lambda: zoom_button(1.25), icon='zoom_in', compound='image').pack(side='left', padx=1)
    ttk.Button(zoom_toolbar, text="Ajustar", command=set_fit_mode).pack(side="left", padx=1)
    ttk.Button(zoom_toolbar, text="Largura", command=set_width_mode).pack(side="left", padx=1)
    ttk.Button(zoom_toolbar, text="100%", command=set_100_mode).pack(side="left", padx=1)
    titlebar.bind('<Configure>', lambda event: preview_title_label.config(
        wraplength=max(80, event.width - zoom_toolbar.winfo_reqwidth() - 8)))

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
    preview_canvas.bind('<Leave>', on_canvas_leave)
    preview_canvas.bind("<Double-Button-1>", on_canvas_double_click)
    preview_canvas.bind("<MouseWheel>", on_canvas_wheel)   # Windows/macOS
    preview_canvas.bind("<Button-4>", on_canvas_wheel)      # Linux, wheel up
    preview_canvas.bind("<Button-5>", on_canvas_wheel)      # Linux, wheel down
    sync_review_radius()

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
    status_row = ttk.Frame(footer)
    status_row.pack(fill='x')
    operation_slot = ttk.Frame(status_row, width=30, height=24)
    operation_slot.pack(side='left')
    operation_slot.pack_propagate(False)
    logo = Image.open(assets_dir / 'riima_logo.png').convert('RGBA')
    logo.thumbnail((20, 20), Image.Resampling.LANCZOS)
    logo_canvas = Image.new('RGBA', (24, 24))
    logo_canvas.alpha_composite(logo, ((24 - logo.width) // 2, (24 - logo.height) // 2))
    operation_frames = [ImageTk.PhotoImage(logo_canvas.rotate(angle, resample=Image.Resampling.BICUBIC), master=root)
                        for angle in range(0, 360, 24)]
    operation_progress = ttk.Label(operation_slot, image=operation_frames[0])
    def stop_operation_indicator():
        if session['operation_after'] is not None:
            root.after_cancel(session['operation_after'])
            session['operation_after'] = None
        if operation_progress.winfo_exists():
            operation_progress.place_forget()
    def animate_operation_indicator():
        session['operation_after'] = None
        if not session['busy'] or not operation_progress.winfo_exists():
            return
        session['operation_frame'] = (session['operation_frame'] + 1) % len(operation_frames)
        operation_progress.config(image=operation_frames[session['operation_frame']])
        session['operation_after'] = root.after(40, animate_operation_indicator)
    def start_operation_indicator():
        stop_operation_indicator()
        operation_progress.place(x=0, y=0, width=24, height=24)
        animate_operation_indicator()
    operation_progress.bind('<Destroy>', lambda event: stop_operation_indicator() if event.widget is operation_progress else None)
    operation_status = ttk.Label(status_row, textvariable=status_var, foreground=theme.TEXT_SECONDARY)
    operation_status.pack(side='left', fill='x', expand=True)
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
