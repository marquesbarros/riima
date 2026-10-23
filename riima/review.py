"""Dataset-wide interactive cluster refinement and resumable projects.

Initial optical labels are immutable. Review labels and approved masks have stable
group IDs independent of the immutable raw cluster provenance. This module contains no Tk code and
never modifies input images.
"""
from __future__ import annotations

from dataclasses import asdict
from copy import deepcopy
import colorsys
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
import tempfile
import zipfile

import cv2
import numpy as np

from . import core
from .exports import export_operation, staged_directory, staged_file

def supervised_backend():
    """Require the OpenCV ML interface used by saved and newly trained forests."""
    ml = getattr(cv2, 'ml', None)
    if not callable(getattr(ml, 'RTrees_create', None)) or not hasattr(ml, 'ROW_SAMPLE'):
        raise RuntimeError(
            'Revisão supervisionada indisponível: cv2.ml.RTrees_create não está disponível. '
            f'OpenCV {getattr(cv2, "__version__", "desconhecido")}; '
            f'módulo: {getattr(cv2, "__file__", "desconhecido")}. '
            f'Interpretador: {sys.executable}. '
            'Reinstale a dependência nesse interpretador: '
            f'"{sys.executable}" -m pip install --upgrade --force-reinstall "opencv-python>=4.8,<5". '
            'Evite instalar simultaneamente distribuições headless ou contrib do OpenCV.')
    return ml
class LocalOpticalSupport:
    """Nearest-neighbour data description in the existing normalized feature space.

    Accept a query only as close to its nearest example as that example is to
    its nearest distinct training neighbour (Tax, 2001, section 3.4.2).
    Also require every coordinate to lie within its observed training range,
    preventing a broad spacing in one attribute from extrapolating another.
    A single unique profile supports exact matches only, without a guessed radius.
    """

    def __init__(self, examples, *, exact=False):
        self.examples = np.ascontiguousarray(np.unique(examples, axis=0), dtype=np.float32)
        self.radius_squared = np.zeros(len(self.examples), np.float32)
        self.minimum = self.examples.min(axis=0) if len(self.examples) else None
        self.maximum = self.examples.max(axis=0) if len(self.examples) else None
        if len(self.examples) > 1 and not exact:
            distances, _ = cv2.batchDistance(self.examples, self.examples, cv2.CV_32F,
                                             normType=cv2.NORM_L2SQR, K=2)
            self.radius_squared = distances[:, 1]

    def contains(self, queries):
        if not len(self.examples):
            return np.zeros(len(queries), bool)
        distances, nearest = cv2.batchDistance(np.ascontiguousarray(queries, dtype=np.float32),
                                               self.examples, cv2.CV_32F,
                                               normType=cv2.NORM_L2SQR, K=1)
        observed = np.all((queries >= self.minimum) & (queries <= self.maximum), axis=1)
        return observed & (distances[:, 0] <= self.radius_squared[nearest[:, 0]])


def prepare_independent_analysis(paths, output, cfg=None, *, mask_mode='auto', stage_callback=None):
    """Pool Lab/T samples from independent fields, without geometric registration."""
    cfg = cfg or core.Config()
    if not paths:
        raise ValueError('Selecione ao menos uma imagem.')
    if not 2 <= cfg.clusters <= 255 or cfg.sample_pixels < cfg.clusters:
        raise ValueError('K deve estar entre 2 e 255 e caber na amostra.')
    if len(paths) * cfg.clusters > cfg.sample_pixels:
        raise ValueError('O orçamento da amostra deve comportar ao menos K pixels por imagem.')
    items, samples = [], []
    rng = np.random.default_rng(cfg.seed)
    # Bound the combined sample, with representation from each field.
    per_image = max(cfg.clusters, cfg.sample_pixels // len(paths))
    for i, path in enumerate(paths):
        if stage_callback:
            stage_callback(f'Extraindo atributos: imagem {i + 1}/{len(paths)}...')
        rgb = core.read_rgb(path)
        overlap = np.ones(rgb.shape[:2], dtype=bool)
        valid = core.analysis_mask_components([rgb], overlap, mask_mode=mask_mode)['valid']
        indices = np.flatnonzero(valid)
        if not len(indices):
            raise ValueError(f'Nenhum pixel válido em {Path(path).name}.')
        sources = core.extract_optical_features([rgb], cfg.feature_blur_sigma)
        chosen = rng.choice(indices, min(per_image, len(indices)), replace=False)
        samples.append(core.collect_features(sources, chosen))
        items.append(dict(id=i, name=Path(path).name, path=str(Path(path).resolve()),
                          rgb=rgb, valid=valid, sources=sources))
    X = np.concatenate(samples)
    if len(X) < cfg.clusters:
        raise ValueError('Pixels válidos insuficientes para o K escolhido.')
    q25, center, q75 = np.percentile(X, [25, 50, 75], axis=0)
    scale = q75 - q25
    scale[scale < 1e-6] = 1
    center, scale = center.astype(np.float32), scale.astype(np.float32)
    X = ((X - center) / scale).astype(np.float32)
    if stage_callback:
        stage_callback('Classificando o conjunto de imagens...')
    _, centers = core.fit_kmeans(X, cfg.clusters, cfg.kmeans_attempts, cfg.seed)
    palette = core.cluster_palette(cfg.clusters)
    for item in items:
        labels, confidence = core.predict_chunked(item['sources'], center, scale, centers, cfg.chunk_pixels)
        labels, confidence = labels.reshape(item['valid'].shape), confidence.reshape(item['valid'].shape)
        edges = core.edge_strength_multi([item['rgb']])
        labels = core.confidence_regularize(labels, confidence, edges, cfg.clusters, cfg, item['valid'])
        labels = core.cleanup_uncertain_islands(labels, confidence, edges, item['valid'], cfg.clusters, cfg)
        optical = dict(labels=labels, valid=item['valid'], confidence=confidence, reference=item['rgb'],
                       images=[item['rgb']], clusters=cfg.clusters, palette=palette,
                       registered_images=[dict(name=item['name'])], image_names=[item['name']],
                       reference_index=0, mask_mode=mask_mode, centers=centers, center=center,
                       scale=scale, sample_X=X, feature_blur_sigma=cfg.feature_blur_sigma,
                       stats=_statistics(labels, item['valid'], confidence, cfg.clusters))
        labels.setflags(write=False)
        item['optical'] = optical
    prepared = dict(output=str(output), config=cfg, optical=items[0]['optical'],
                    interpretation=core.Interpretation(items[0]['optical']),
                    dataset_items=items, analysis_mode='independent')
    prepared['review'] = ClusterReview.from_prepared(prepared)
    return prepared


def _statistics(labels, valid, confidence, k):
    total = max(1, int(valid.sum()))
    return [dict(cluster=c, pixels=int((valid & (labels == c)).sum()),
                 area=100 * int((valid & (labels == c)).sum()) / total,
                 confidence=float(confidence[valid & (labels == c)].mean())
                 if np.any(valid & (labels == c)) else 0.) for c in range(k)]


class ClusterReview:
    """One shared set of classes/models over one or many pixel grids."""

    @classmethod
    def from_prepared(cls, prepared):
        optical = prepared['optical']
        state = cls()
        state.mode = prepared.get('analysis_mode', 'registered')
        state.config = prepared['config']
        state.k = int(optical['clusters'])
        state.groups = {c: {c} for c in range(state.k)}
        state.group_names, state.retired_groups = {}, {}
        state.next_group_id = state.k
        state._group_undo, state._catalog_edited = [], False
        state._legacy_group_names = {}
        state._interpretation = prepared['interpretation']
        state.centers = np.asarray(optical['centers'], np.float32).copy()
        state.center = np.asarray(optical['center'], np.float32).copy()
        state.scale = np.asarray(optical['scale'], np.float32).copy()
        state.palette = np.asarray(optical['palette'], np.uint8).copy()
        state.completed, state.models, state.items, state.dirty = set(), {}, [], set()
        state.refinement_order = []
        state.exclusion_dirty = False
        state.unresolved_dirty = set()
        state.background_groups, state.applied_background_groups = set(), set()
        state.group_policy_dirty = False
        state._undo, state._stroke = [], None
        state.selected_image, state.selected_cluster = 0, 0
        if state.mode == 'independent':
            source_items = prepared['dataset_items']
        else:
            # Registered views share one pixel grid and a single 4N feature vector.
            source_items = [dict(id=0, name='Grade registrada', path='', rgb=optical['reference'], optical=optical)]
        for entry in source_items:
            raw = entry['optical']
            initial = raw['labels'].copy()
            initial.setflags(write=False)
            state.items.append(dict(id=entry['id'], name=entry['name'], path=entry['path'],
                                    rgb=entry['rgb'], optical=raw, initial=initial,
                                    labels=initial.astype(np.int32), valid=raw['valid'].copy(),
                                    locked=np.full(initial.shape, -1, np.int32), annotations={},
                                    applied_annotations={}, excluded=np.zeros(initial.shape, bool),
                                    applied_excluded=np.zeros(initial.shape, bool),
                                    display_valid=raw['valid'].copy(),
                                    background_masks={},
                                    sources=entry.get('sources')))
        state.sync_groups(prepared.get('review_groups', prepared['interpretation'].groups))
        interpretation = prepared['interpretation']
        for index, members in enumerate(interpretation.groups):
            group = min(members)
            if state.groups.get(group) == members:
                if index in interpretation.names:
                    state._legacy_group_names[group] = interpretation.names[index]
                if index in interpretation.colors:
                    state.palette[group] = interpretation.colors[index]
        return state

    @property
    def class_ids(self):
        return sorted(self.groups)

    def group_name(self, group, interpretation=None):
        """Resolve current names by stable ID, with legacy raw-partition fallback."""
        self._cluster(group)
        if group in self.group_names:
            return self.group_names[group] or f'Grupo {group}'
        interpretation = interpretation if interpretation is not None else self._interpretation
        if interpretation is not None and self.groups[group]:
            for index, members in enumerate(interpretation.groups):
                if members == self.groups[group]:
                    return interpretation.names.get(index) or f'Grupo {group}'
        return self._legacy_group_names.get(group) or f'Grupo {group}'

    @property
    def can_undo_group(self):
        return bool(self._group_undo)

    def _group_snapshot(self):
        # Scientific inputs are immutable shared session data, never history.
        excluded = {'items', 'config', 'k', 'centers', 'center', 'scale', '_group_undo', '_interpretation'}
        metadata = {key: (dict(value) if key == 'models' else deepcopy(value))
                    for key, value in self.__dict__.items() if key not in excluded}
        item_keys = ('labels', 'locked', 'annotations', 'applied_annotations', 'excluded',
                     'applied_excluded', 'display_valid', 'background_masks')
        return metadata, [{key: deepcopy(item[key]) for key in item_keys} for item in self.items]

    def _restore_group_snapshot(self, snapshot, *, retain_ids=False):
        high_water, palette = self.next_group_id, self.palette
        metadata, items = snapshot
        for key in ('refinement_evidence',):
            if key not in metadata:
                self.__dict__.pop(key, None)
        self.__dict__.update(metadata)
        for item, saved in zip(self.items, items):
            item.update(saved)
        if retain_ids and high_water > self.next_group_id:
            self.next_group_id = high_water
            if len(self.palette) < high_water:
                self.palette = np.concatenate((self.palette, palette[len(self.palette):high_water]))
            self._catalog_edited = True

    def _remember_group(self, snapshot):
        self._group_undo.append(snapshot)
        del self._group_undo[:-10]
        # Keep at most 128 MiB of NumPy history, except the latest reversible op.
        def size(value):
            if isinstance(value, np.ndarray):
                return value.nbytes
            if isinstance(value, dict):
                return sum(size(v) for v in value.values())
            if isinstance(value, (tuple, list)):
                return sum(size(v) for v in value)
            return 0
        while len(self._group_undo) > 1 and size(self._group_undo) > 128 * 1024 ** 2:
            self._group_undo.pop(0)
        self._catalog_edited = True
        self._undo, self._stroke = [], None

    def undo_group_operation(self):
        """Restore the state before a group edit, including intervening review edits.

        This bounded session history is separate from individual brush-stroke undo.
        Successful group IDs remain consumed even when creation is undone.
        """
        if not self._group_undo:
            return False
        self._restore_group_snapshot(self._group_undo.pop(), retain_ids=True)
        return True

    def create_group(self, name='', color=None):
        group = self.next_group_id
        if group > min(np.iinfo(np.int32).max - self.config.seed, np.iinfo(np.int32).max - 1):
            raise ValueError('Limite de identificadores de grupos atingido.')
        if color is None:
            candidates = np.array([tuple(round(v * 255) for v in colorsys.hsv_to_rgb(i / 360, s, v))
                                   for s in (.55, .85) for v in (.72, 1.) for i in range(360)], np.int32)
            active = self.palette[self.class_ids].astype(np.int32)
            distances = np.min(np.sum((candidates[:, None] - active[None]) ** 2, axis=2), axis=1)
            color = candidates[int(np.argmax(distances))]
        color = core.Interpretation._color(color)
        snapshot = self._group_snapshot()
        self.groups[group] = set()
        self.group_names[group] = str(name).strip()
        self.palette = np.concatenate((self.palette, np.array([color], np.uint8)))
        self.next_group_id += 1
        self._remember_group(snapshot)
        return group

    def rename_group(self, group, name):
        self._cluster(group)
        name = str(name).strip()
        if self.group_names.get(group) == name and group in self.group_names:
            return
        snapshot = self._group_snapshot()
        self.group_names[group] = name
        self._remember_group(snapshot)

    def recolor_group(self, group, color):
        self._cluster(group)
        color = core.Interpretation._color(color)
        if np.array_equal(self.palette[group], color):
            return
        snapshot = self._group_snapshot()
        self.palette[group] = color
        self._remember_group(snapshot)

    def _retire_group(self, group, reason):
        self.retired_groups[group] = dict(members=sorted(self.groups[group]), name=self.group_name(group),
                                          color=self.palette[group].tolist(), reason=reason)
        del self.groups[group]
        self.group_names.pop(group, None)
        self._legacy_group_names.pop(group, None)
        self.models.pop(group, None)
        for key in ('completed', 'dirty', 'unresolved_dirty', 'background_groups', 'applied_background_groups'):
            getattr(self, key).discard(group)
        self.refinement_order = [c for c in self.refinement_order if c != group]
        if hasattr(self, 'refinement_evidence'):
            self.refinement_evidence.pop(group, None)
        for item in self.items:
            for key in ('annotations', 'applied_annotations', 'background_masks'):
                item[key].pop(group, None)

    def merge_groups(self, ids):
        selected = sorted(set(ids))
        if len(selected) < 2:
            raise ValueError('Selecione ao menos dois grupos para unir.')
        for group in selected:
            self._cluster(group, editable=True)
        if any(c in self.background_groups for c in selected):
            raise ValueError('Restaure os grupos de fundo antes de unir.')
        target = selected[0]
        snapshot = self._group_snapshot()
        try:
            self.group_names.setdefault(target, self.group_name(target))
            reviewed = any(c in self.models or c in self.dirty or c in self.unresolved_dirty
                           or any(np.any(item[key].get(c, 0)) for item in self.items
                                  for key in ('annotations', 'applied_annotations')) for c in selected)
            members = set().union(*(self.groups[c] for c in selected))
            for item in self.items:
                item['labels'][np.isin(item['labels'], selected)] = target
                for key in ('annotations', 'applied_annotations'):
                    if not any(c in item[key] for c in selected):
                        continue
                    merged = np.zeros(item['valid'].shape, np.int8)
                    rejected_by_all = np.ones(item['valid'].shape, bool)
                    for group in selected:
                        marks = item[key].get(group)
                        if marks is None:
                            rejected_by_all.fill(False)
                        else:
                            rejected_by_all &= marks == -1
                            merged[marks == 1] = 1
                    merged[rejected_by_all] = -1
                    item[key][target] = merged
            for group in selected[1:]:
                self._retire_group(group, f'merged_into:{target}')
            self.groups[target] = members
            self.models.pop(target, None)
            if reviewed:
                self.dirty.add(target)
            if self.selected_cluster in selected:
                self.selected_cluster = target
        except Exception:
            self._restore_group_snapshot(snapshot)
            raise
        self._remember_group(snapshot)
        return target

    def delete_group(self, group):
        self._cluster(group, editable=True)
        if len(self.groups) <= 1:
            raise ValueError('Não é possível excluir o último grupo interpretativo disponível.')
        filled = any(np.any(item['valid'] & (item['labels'] == group)) for item in self.items)
        if filled and self.has_pending:
            raise ValueError('Atualize a classificação para aplicar as marcações pendentes antes de excluir.')
        if group in self.background_groups:
            raise ValueError('Restaure o grupo de fundo antes de excluir.')
        snapshot = self._group_snapshot()
        try:
            self._retire_group(group, 'deleted')
            if filled:
                self._predict()
                for item in self.items:
                    eligible = item['valid'] & ~item['excluded']
                    if np.any(eligible & ~np.isin(item['labels'], self.class_ids)):
                        raise ValueError('As restrições impedem uma reclassificação completa nos grupos de destino.')
            if self.selected_cluster == group:
                self.selected_cluster = self.class_ids[0]
        except Exception:
            self._restore_group_snapshot(snapshot)
            raise
        self._remember_group(snapshot)

    @property
    def has_pending(self):
        return bool(self.dirty or self.exclusion_dirty or self.group_policy_dirty)

    @property
    def can_undo(self):
        return bool(self._undo)

    def begin_stroke(self):
        self.end_stroke()
        self._stroke = []

    def end_stroke(self):
        if self._stroke:
            self._undo.append(self._stroke)
            del self._undo[:-50]
        self._stroke = None

    def _record_patch(self, image_id, key, cluster, indices, previous):
        patch = (image_id, key, cluster, indices.copy(), previous.copy())
        if self._stroke is None:
            self._undo.append([patch])
            del self._undo[:-50]
        else:
            self._stroke.append(patch)

    def undo(self):
        """Restore a complete brush stroke; prediction still waits for Update."""
        self.end_stroke()
        if not self._undo:
            return False
        for image_id, key, cluster, indices, previous in reversed(self._undo.pop()):
            item = self.items[image_id]
            buffer = item[key][cluster] if key in ('annotations', 'background_masks') else item[key]
            buffer.ravel()[indices] = previous
        self.dirty = {c for c in self.class_ids if c not in self.completed and any(
            (not np.array_equal(item['annotations'][c], item['applied_annotations'][c])
             if c in item['annotations'] and c in item['applied_annotations'] else
             bool(np.any(item['annotations'].get(c, 0))) or bool(np.any(item['applied_annotations'].get(c, 0))))
            for item in self.items)}
        # Older projects had no applied-example snapshot. Their pending state is
        # authoritative until Update, even when all examples have been erased.
        self.dirty |= self.unresolved_dirty
        self.exclusion_dirty = any(not np.array_equal(item['excluded'], item['applied_excluded'])
                                   for item in self.items)
        return True

    def initial_labels(self, item):
        lut = np.full(self.k, -1, np.int32)
        for group, members in self.groups.items():
            lut[list(members)] = group
        return lut[item['initial']]

    def sync_groups(self, groups):
        """Adopt the interpretation partition without destroying existing reviews."""
        partition = core._validate_partition([set(g) for g in groups], self.k)
        incoming = {min(g): set(g) for g in partition}
        if self._catalog_edited and incoming != self.groups:
            raise ValueError('O catálogo atual usa identidades estáveis. Use as operações de grupos para reorganizá-lo.')
        unchanged = {c for c, members in self.groups.items() if incoming.get(c) == members}
        changed = set(self.groups) - unchanged
        for c in changed:
            reviewed = (c in self.completed or c in self.models or c in self.background_groups
                        or c in self.applied_background_groups or c in self.dirty or c in self.unresolved_dirty
                        or any(np.any(item[key].get(c, np.zeros(0)))
                               for item in self.items for key in ('annotations', 'applied_annotations')))
            if reviewed:
                raise ValueError('Esse grupo já tem revisão. Preserve seu agrupamento ou reabra, apague suas marcações e clique em Atualizar classificação antes de reagrupar.')
        if incoming != self.groups:
            self._undo, self._stroke = [], None
            lut = np.empty(self.k, np.int32)
            for group, members in incoming.items():
                lut[list(members)] = group
            for item in self.items:
                previous = item['labels'].copy()
                for old in changed:
                    affected = item['valid'] & (previous == old)
                    children = {group for group, members in incoming.items() if members & self.groups[old]}
                    if len(children) == 1:
                        item['labels'][affected] = next(iter(children))
                    else:
                        mapped = lut[item['initial'][affected]]
                        # A fallback pixel may originate outside the group being split.
                        # Keep it among this group's children, never in another approved ID.
                        mapped[~np.isin(mapped, list(children))] = min(children)
                        item['labels'][affected] = mapped
                for c in changed:
                    item['annotations'].pop(c, None)
                    item['applied_annotations'].pop(c, None)
            self.dirty -= changed
            self.unresolved_dirty -= changed
            self.groups = incoming
            if self.selected_cluster not in self.groups:
                self.selected_cluster = self.class_ids[0]

    def _cluster(self, cluster, *, editable=False):
        if cluster not in self.groups:
            raise ValueError('Cluster inexistente.')
        if editable and cluster in self.completed:
            raise ValueError('Reabra o cluster pronto antes de editar.')

    def _sources(self, item):
        if item['sources'] is None:
            images = item['optical']['images'] if self.mode == 'registered' else [item['rgb']]
            item['sources'] = core.extract_optical_features(images, self.config.feature_blur_sigma)
        return item['sources']

    def _features(self, item, indices):
        return ((core.collect_features(self._sources(item), indices) - self.center) / self.scale).astype(np.float32)

    def annotate(self, image_id, cluster, indices, value):
        """Set positive (+1), negative (-1) or erase (0) examples on valid unlocked pixels."""
        self._cluster(cluster, editable=True)
        if value not in (-1, 0, 1):
            raise ValueError('Marcação inválida.')
        item = self.items[image_id]
        indices = np.unique(np.asarray(indices, dtype=np.int64))
        indices = indices[(indices >= 0) & (indices < item['valid'].size)]
        allowed = item['valid'].ravel()[indices] & (item['locked'].ravel()[indices] < 0)
        if value:
            allowed &= ~item['excluded'].ravel()[indices]
        indices = indices[allowed]
        if value == 1:
            for other, marks in item['annotations'].items():
                if other != cluster and np.any(marks.ravel()[indices] == 1):
                    raise ValueError('Há exemplos positivos de outro cluster. Apague a marcação anterior primeiro.')
        marks = item['annotations'].get(cluster)
        if marks is None:
            marks = item['annotations'][cluster] = np.zeros(item['valid'].shape, np.int8)
        changed = indices[marks.ravel()[indices] != value]
        if len(changed):
            self._record_patch(image_id, 'annotations', cluster, changed, marks.ravel()[changed])
            self.dirty.add(cluster)
            marks.ravel()[changed] = value

    def exclude(self, image_id, indices, value=True, *, group_action=False):
        """Mark slide damage/background globally, without changing any approved pixel."""
        item = self.items[image_id]
        indices = np.unique(np.asarray(indices, dtype=np.int64))
        indices = indices[(indices >= 0) & (indices < item['valid'].size)]
        allowed = item['valid'].ravel()[indices] & (item['locked'].ravel()[indices] < 0)
        if not group_action:
            # A manual stroke takes ownership, even when the excluded value was
            # already True. Resetting/restoring a group cannot erase that choice.
            for c, mask in item['background_masks'].items():
                detached = indices[allowed & mask.ravel()[indices]]
                if len(detached):
                    self._record_patch(image_id, 'background_masks', c, detached, mask.ravel()[detached])
                    mask.ravel()[detached] = False
        changed = indices[allowed & (item['excluded'].ravel()[indices] != bool(value))]
        if len(changed):
            self._record_patch(image_id, 'excluded', None, changed, item['excluded'].ravel()[changed])
            item['excluded'].ravel()[changed] = bool(value)
            self.exclusion_dirty = True

    def set_background(self, clusters, value=True):
        """Send complete current masks to background, or restore their recorded areas."""
        clusters = set(clusters)
        for c in clusters:
            self._cluster(c, editable=True)
        for c in clusters:
            if value and c not in self.background_groups:
                for i, item in enumerate(self.items):
                    mask = item['valid'] & (item['labels'] == c) & (item['locked'] < 0) & ~item['excluded']
                    item['background_masks'][c] = mask.copy()
                    self.exclude(i, np.flatnonzero(mask), True, group_action=True)
                self.background_groups.add(c)
            elif not value and c in self.background_groups:
                for i, item in enumerate(self.items):
                    mask = item['background_masks'].pop(c, None)
                    if mask is not None:
                        # Preserve ownership of another background group too.
                        for other in item['background_masks'].values():
                            mask &= ~other
                        self.exclude(i, np.flatnonzero(mask), False, group_action=True)
                self.background_groups.remove(c)
        self.group_policy_dirty = self.background_groups != self.applied_background_groups
        # Group-wide actions have their own explicit inverse, Restore selection.
        self._undo, self._stroke = [], None

    def reset_group(self, cluster):
        """Clear this group's examples/model at Update, preserving other masks and background."""
        self._cluster(cluster, editable=True)
        if cluster in self.background_groups:
            self.set_background([cluster], False)
        for item in self.items:
            marks = item['annotations'].get(cluster)
            if marks is not None:
                marks.fill(0)
        self.dirty.add(cluster)
        self._undo, self._stroke = [], None

    def _training_examples(self, cluster, rng=None, *, all_profiles=False):
        """Sample forest training rows, or collect every distinct annotated profile.

        Contradiction detection must see all evidence before the forest's brush
        budget is applied. Build that inventory in chunks without materializing
        another full set of feature planes or all repeated annotated rows.
        """
        examples, responses = [], []
        rng = rng if rng is not None else np.random.default_rng(self.config.seed + cluster)
        profiles = {value: np.empty((0, len(self.center)), np.float32) for value in (-1, 1)}
        for item in self.items:
            marks = item['annotations'].get(cluster)
            if marks is None:
                continue
            for value in (-1, 1):
                idx = np.flatnonzero((marks == value) & item['valid'] & ~item['excluded'] & (item['locked'] < 0))
                if all_profiles:
                    for start in range(0, len(idx), self.config.chunk_pixels):
                        chunk = self._features(item, idx[start:start + self.config.chunk_pixels])
                        profiles[value] = np.unique(np.concatenate((profiles[value], np.unique(chunk, axis=0))), axis=0)
                    continue
                if len(idx) > 5000:
                    idx = rng.choice(idx, 5000, replace=False)
                if len(idx):
                    examples.append(self._features(item, idx))
                    responses.append(np.full(len(idx), int(value == 1), np.int32))
        if all_profiles:
            for value in (-1, 1):
                if len(profiles[value]):
                    examples.append(profiles[value])
                    responses.append(np.full(len(profiles[value]), int(value == 1), np.int32))
        if not responses:
            return np.empty((0, len(self.center)), np.float32), np.empty(0, np.int32)
        return np.concatenate(examples), np.concatenate(responses)

    @staticmethod
    def _conflicting_examples(X, y):
        unique, inverse = np.unique(X, axis=0, return_inverse=True)
        positive = np.bincount(inverse[y == 1], minlength=len(unique)) > 0
        negative = np.bincount(inverse[y == 0], minlength=len(unique)) > 0
        conflict = positive & negative
        return unique[conflict], conflict[inverse]

    def _fit_model(self, cluster):
        """Fit a binary forest only from optically separable positive/negative evidence."""
        self._cluster(cluster, editable=True)
        rng = np.random.default_rng(self.config.seed + cluster)
        profiles, profile_responses = self._training_examples(cluster, all_profiles=True)
        X, y = self._training_examples(cluster, rng)
        if len(y):
            conflicts, _ = self._conflicting_examples(profiles, profile_responses)
            if len(conflicts):
                # Structured numeric records compare rows exactly (including
                # equivalent signed zeros) without a quadratic distance matrix.
                row_dtype = np.dtype([(str(i), X.dtype) for i in range(X.shape[1])])
                conflict = np.isin(np.ascontiguousarray(X).view(row_dtype).ravel(),
                                   np.ascontiguousarray(conflicts).view(row_dtype).ravel())
                X, y = X[~conflict], y[~conflict]
        if len(np.unique(y)) == 2:
            # Balance brush sizes, preserving every sampled explicit example.
            positive, negative = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
            n = max(len(positive), len(negative), 32)
            train_idx = np.concatenate([positive, rng.choice(positive, n - len(positive), replace=True),
                                        negative, rng.choice(negative, n - len(negative), replace=True)])
            ml = supervised_backend()
            cv2.setRNGSeed(self.config.seed + cluster)
            model = ml.RTrees_create()
            model.setMaxDepth(12)
            model.setMinSampleCount(1)
            model.setActiveVarCount(X.shape[1])
            model.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 64, 0))
            if not model.train(X[train_idx], ml.ROW_SAMPLE, y[train_idx]):
                raise ValueError('Não foi possível treinar o refinamento.')
            # Insertion order records the latest successfully refined model.
            self.models.pop(cluster, None)
            self.models[cluster] = model
        else:
            # Erasing examples must discard stale learned decisions.
            self.models.pop(cluster, None)
        return cluster in self.models

    def refine(self, cluster):
        self._cluster(cluster, editable=True)
        previous_models = dict(self.models)
        previous_order = list(self.refinement_order)
        try:
            trained = self._fit_model(cluster)
            self.refinement_order = [c for c in previous_order if c != cluster] + [cluster]
            self._predict(preferred=cluster)
        except Exception:
            self.models = previous_models
            self.refinement_order = previous_order
            raise
        self._commit_examples([cluster])
        return trained

    def _commit_examples(self, clusters):
        for item in self.items:
            for c in clusters:
                if c in item['annotations']:
                    item['applied_annotations'][c] = item['annotations'][c].copy()
            item['applied_excluded'] = item['excluded'].copy()
            item['display_valid'] = item['valid'] & ~item['applied_excluded']
        self.dirty.difference_update(clusters)
        self.unresolved_dirty.difference_update(clusters)
        self.exclusion_dirty = False
        self.applied_background_groups = set(self.background_groups)
        self.group_policy_dirty = False

    def refine_pending(self, preferred=None):
        """Apply the user's accumulated changes with a single dataset prediction."""
        pending = sorted(self.dirty - self.completed)
        edited = set(pending)
        if not self.has_pending:
            return {}
        if self.exclusion_dirty or self.group_policy_dirty:
            # Global exclusions change the training samples of all open groups.
            pending = [c for c in self.class_ids if c not in self.completed]
        previous_models = dict(self.models)
        previous_order = list(self.refinement_order)
        model_order = list(previous_models)
        try:
            trained = {c: self._fit_model(c) for c in pending if c not in self.background_groups}
            if self.exclusion_dirty or self.group_policy_dirty:
                # Re-training for background alone must preserve learned precedence.
                self.models = {c: self.models[c] for c in dict.fromkeys(model_order + list(self.models))
                               if c in self.models}
            self.refinement_order = [c for c in previous_order if c not in edited] + sorted(edited)
            self._predict(preferred=preferred if preferred in edited else None)
        except Exception:
            self.models = previous_models
            self.refinement_order = previous_order
            raise
        self._commit_examples(pending)
        return trained

    def _predict(self, preferred=None):
        open_ids = [c for c in self.class_ids if c not in self.completed and c not in self.background_groups]
        supports, conflicts, evidence = {}, {}, {}
        for c in open_ids:
            X, y = self._training_examples(c, all_profiles=True)
            if not len(y):
                continue
            conflict, conflicting_rows = self._conflicting_examples(X, y)
            if len(conflict):
                conflicts[c] = LocalOpticalSupport(conflict, exact=True)
                evidence[c] = 'conflicting'
            clean_X, clean_y = X[~conflicting_rows], y[~conflicting_rows]
            if len(np.unique(clean_y)) == 1:
                unique = np.unique(clean_X, axis=0)
                if len(unique) > 5000:
                    rng = np.random.default_rng(self.config.seed + c)
                    unique = unique[rng.choice(len(unique), 5000, replace=False)]
                supports[c] = (int(clean_y[0]), LocalOpticalSupport(unique))
                evidence[c] = ('positive' if clean_y[0] else 'negative') + ('_conflicting' if len(conflict) else '')
        # Both evidence strategies use the same successful-update precedence.
        # Missing historical entries retain support-before-forest semantics.
        fallback = [c for c in supports if c not in self.models] + list(self.models)
        actions = set(supports) | set(self.models)
        marked = {c for item in self.items for c, marks in item['annotations'].items() if np.any(marks)}
        next_order = [c for c in fallback if c not in self.refinement_order] + self.refinement_order
        # Exclusions can temporarily hide all examples. Preserve their priority
        # for restoration; erase/reset without remaining evidence removes it.
        next_order = [c for c in next_order if c in self.groups and
                      (c not in open_ids or c in actions or c in marked)]
        if preferred in actions and preferred in open_ids:
            next_order = [c for c in next_order if c != preferred] + [preferred]
        order = [c for c in next_order if c in open_ids]
        updates = []
        for item in self.items:
            available = item['valid'] & ~item['excluded'] & (item['locked'] < 0)
            indices = np.flatnonzero(available)
            result = item['labels'].copy()
            result[~item['valid'] | item['excluded']] = -1
            baseline = self.initial_labels(item).ravel()
            for start in range(0, len(indices), self.config.chunk_pixels):
                idx = indices[start:start + self.config.chunk_pixels]
                X = self._features(item, idx)
                labels = np.full(len(idx), -1, np.int32)
                if open_ids:
                    best = np.full(len(idx), np.inf, np.float32)
                    previous = baseline[idx]
                    retained = np.zeros(len(idx), bool)
                    predictions = {}
                    support_matches = {c: model.contains(X) for c, (_, model) in supports.items()}
                    for c in supports.keys() & conflicts.keys():
                        support_matches[c] &= ~conflicts[c].contains(X)
                    positive_claims = sum((support_matches[c].astype(np.int32) for c in supports if supports[c][0]),
                                          start=np.zeros(len(idx), np.int32))
                    for c in open_ids:
                        d = np.full(len(idx), np.inf, np.float32)
                        for raw_id in self.groups[c]:
                            d = np.minimum(d, np.sum((X - self.centers[raw_id]) ** 2, axis=1))
                        allowed = np.ones(len(idx), bool)
                        if c in supports and not supports[c][0]:
                            allowed &= ~support_matches[c]
                        marks = item['annotations'].get(c)
                        if marks is not None:
                            allowed &= marks.ravel()[idx] != -1
                        model = self.models.get(c)
                        if model is not None:
                            _, predicted = model.predict(X)
                            predictions[c] = predicted.ravel() == 1
                            if c in conflicts:
                                ambiguous = conflicts[c].contains(X)
                                predictions[c] &= ~ambiguous
                                allowed &= predictions[c] | ambiguous
                            else:
                                allowed &= predictions[c]
                        d = np.where(allowed, d, np.inf)
                        retained |= (previous == c) & allowed
                        take = d < best
                        best[take], labels[take] = d[take], c
                    # Keep the current spatially regularized selection where admissible.
                    labels[retained] = previous[retained]
                    for c in order:
                        take = predictions.get(c, np.zeros(len(idx), bool)).copy()
                        if c in supports and supports[c][0]:
                            take |= support_matches[c] & (positive_claims == 1)
                        marks = item['annotations'].get(c)
                        if marks is not None:
                            take &= marks.ravel()[idx] != -1
                        labels[take] = c
                    for c in open_ids:
                        marks = item['annotations'].get(c)
                        if marks is not None:
                            labels[marks.ravel()[idx] == 1] = c
                result.ravel()[idx] = labels
            frozen = item['locked'] >= 0
            result[frozen] = item['locked'][frozen]
            updates.append(result)
        for item, result in zip(self.items, updates):
            item['labels'] = result
        self.refinement_evidence = evidence
        self.refinement_order = next_order
        if preferred in self.models and preferred in open_ids:
            # Persist the effective conflict order for future updates and save/resume.
            self.models[preferred] = self.models.pop(preferred)

    def approve(self, cluster):
        self._cluster(cluster, editable=True)
        if cluster in self.background_groups:
            raise ValueError('O grupo está marcado como fundo. Restaure-o antes de aprovar.')
        # Approval freezes the displayed result and never launches hidden computation.
        if self.has_pending:
            raise ValueError('Atualize a classificação para aplicar as marcações pendentes antes de aprovar.')
        if not any(np.any(item['valid'] & (item['labels'] == cluster)) for item in self.items):
            raise ValueError('O cluster está vazio; revise os exemplos antes de aprovar.')
        for item in self.items:
            mask = item['valid'] & (item['labels'] == cluster) & (item['locked'] < 0)
            item['locked'][mask] = cluster
        self.completed.add(cluster)
        self._undo, self._stroke = [], None

    def reopen(self, cluster):
        self._cluster(cluster)
        if cluster not in self.completed:
            raise ValueError('O cluster ainda está em revisão.')
        self.completed.remove(cluster)
        self._undo, self._stroke = [], None
        for item in self.items:
            item['locked'][item['locked'] == cluster] = -1
        # Preserve the accepted mask until the user explicitly refines again.

    def counts(self, cluster):
        return sum(int((item['valid'] & (item['labels'] == cluster)).sum()) for item in self.items)

    def view(self, image_id=0):
        item = self.items[image_id if self.mode == 'independent' else 0]
        return dict(labels=item['labels'], locked=item['locked'], annotations=item['annotations'],
                    applied_annotations=item['applied_annotations'],
                    excluded=item['excluded'], applied_excluded=item['applied_excluded'],
                    analysis_valid=item['valid'], valid=item['display_valid'],
                    palette=self.palette, completed=self.completed,
                    background_groups=self.background_groups,
                    groups=self.groups, class_ids=self.class_ids)


def _model_text(model):
    storage = cv2.FileStorage('', cv2.FILE_STORAGE_WRITE | cv2.FILE_STORAGE_MEMORY)
    model.write(storage, 'model')
    return storage.releaseAndGetString()


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Valor não serializável: {type(value).__name__}')


def _validate_project_config(cfg):
    integer_fields = ('clusters', 'sample_pixels', 'kmeans_attempts', 'chunk_pixels', 'seed')
    if (any(type(getattr(cfg, key)) is not int for key in integer_fields)
            or not 2 <= cfg.clusters <= 255 or cfg.sample_pixels < cfg.clusters
            or cfg.kmeans_attempts < 1 or cfg.chunk_pixels < 1
            or not 0 <= cfg.seed <= np.iinfo(np.int32).max - (cfg.clusters - 1)
            or type(cfg.feature_blur_sigma) not in (int, float)
            or not np.isfinite(cfg.feature_blur_sigma) or cfg.feature_blur_sigma < 0):
        raise ValueError('Configuração de classificação inválida no projeto.')


def _validate_feature_array(array, shape):
    """Check stored numeric data before the predictor converts it to float32."""
    if (array.dtype.kind not in 'fiu' or array.ndim != len(shape)
            or any(size is not None and actual != size for actual, size in zip(array.shape, shape))
            or not np.all(np.isfinite(array))
            or np.any(np.abs(array.astype(np.float64)) > np.finfo(np.float32).max)):
        raise ValueError('Atributos ópticos inválidos ou incompatíveis no projeto.')


_OPTIONAL_OPTICAL_ARRAYS = ('feature_means', 'overlap', 'palette')


def _validate_workspace(workspace, class_ids):
    """Validate stored controls; missing fields retain their historical GUI defaults.

    Opacity is a percentage in [0, 100]; the split fraction is in [0, 1].
    Obsolete group selections are discarded. Unused extension fields are preserved.
    """
    if not isinstance(workspace, dict):
        raise ValueError('Workspace inválido: esperado um objeto de configurações.')
    result = dict(workspace)
    for field, maximum in (('opacity', 100), ('split_fraction', 1)):
        if field in result:
            value = result[field]
            if type(value) not in (int, float) or not np.isfinite(value) or not 0 <= value <= maximum:
                raise ValueError(f'Workspace inválido: {field} deve estar entre 0 e {maximum}.')
    for field in ('isolate', 'contours', 'show_applied', 'automatic_output'):
        if field in result and type(result[field]) is not bool:
            raise ValueError(f'Workspace inválido: {field} deve ser booleano.')
    choices = dict(step=('workbench', 'preparation', 'interpretation', 'images', 'classify', 'organize', 'refine', 'results'),
                   display=('Imagem', 'Sobreposição', 'Classificação', 'Máscara', 'Máscara selecionada', 'Grupo selecionado'),
                   view_type=('layers', 'compare'))
    for field, allowed in choices.items():
        if field in result and (not isinstance(result[field], str) or result[field] not in allowed):
            raise ValueError(f'Workspace inválido: opção desconhecida em {field}.')
    for field in ('compare_left', 'compare_right'):
        if field in result and not isinstance(result[field], str):
            raise ValueError(f'Workspace inválido: {field} deve ser texto.')
    if 'selected_groups' in result:
        selected = result['selected_groups']
        if not isinstance(selected, list) or any(type(group) is not int for group in selected):
            raise ValueError('Workspace inválido: selected_groups deve ser uma lista de grupos inteiros.')
        result['selected_groups'] = list(dict.fromkeys(group for group in selected if group in class_ids))
    return result


def _load_optional_optical_arrays(archive, index, info, valid, clusters, dimensions):
    """Validate optional data before constructing the optical state."""
    keys = info.get('optional_arrays', [])
    if (not isinstance(keys, list) or any(not isinstance(key, str) for key in keys)
            or len(set(keys)) != len(keys) or any(key not in _OPTIONAL_OPTICAL_ARRAYS for key in keys)):
        raise ValueError('Lista de arrays opcionais inválida: somente feature_means, overlap e palette são permitidos.')
    arrays = {}
    for key in keys:
        array = archive[f'{index}_{key}'].copy()
        if key == 'feature_means':
            _validate_feature_array(array, (clusters, dimensions))
        elif key == 'palette' and (array.dtype != np.uint8 or array.shape != (clusters, 3)):
            raise ValueError('Paleta óptica inválida no projeto.')
        elif key == 'overlap' and (array.dtype != bool or array.shape != valid.shape
                                   or np.any(valid & ~array)):
            raise ValueError('Máscara de sobreposição inválida no projeto.')
        arrays[key] = array
    return arrays


def _project_output_history(prepared):
    """Keep historical paths as provenance, never as a resumed destination."""
    history = prepared.get('output_history', [])
    if (not isinstance(history, list) or any(not isinstance(path, str) or not path for path in history)
            or not isinstance(prepared['output'], str) or not prepared['output']):
        raise ValueError('Histórico de destinos inválido no projeto.')
    return list(dict.fromkeys([*history, prepared['output']]))


def _portable_output(path, history=()):
    """Suggest an unused sibling directory without writing during project loading."""
    path = Path(path).resolve()
    base = path.with_name(path.stem + '_output')
    target, suffix = base, 2
    historical = {os.path.normcase(os.path.abspath(entry)) for entry in history}
    while target.exists() or target.is_symlink() or os.path.normcase(str(target)) in historical:
        target = base.with_name(f'{base.name}_{suffix}')
        suffix += 1
    return str(target)


def _compact_project_arrays(arrays):
    """Store exact duplicates once and narrow only losslessly bounded integers."""
    stored, aliases, integer_dtypes, seen = {}, {}, {}, {}
    # Keep review labels canonical when initial and current classifications agree.
    keys = sorted(arrays, key=lambda key: (not key.endswith('_labels'), key))
    for key in keys:
        array = np.asarray(arrays[key])
        if (key.endswith(('_initial', '_labels', '_locked')) and array.dtype.kind in 'iu'
                and array.dtype.itemsize > 2 and array.size
                and np.min(array) >= -1 and np.max(array) <= 254):
            integer_dtypes[key] = array.dtype.str
            array = array.astype(np.int16)
        contiguous = np.ascontiguousarray(array)
        fingerprint = (array.dtype.str, array.shape,
                       hashlib.sha256(memoryview(contiguous).cast('B')).digest())
        previous = seen.get(fingerprint)
        if (previous is not None
                and np.array_equal(contiguous.view(np.uint8), np.ascontiguousarray(stored[previous]).view(np.uint8))):
            aliases[key] = previous
        else:
            seen[fingerprint] = key
            stored[key] = array
    return stored, aliases, integer_dtypes


class _ProjectArrays:
    """Expose logical arrays without loading every compressed member into memory."""

    def __init__(self, archive, meta):
        self.archive = archive
        self.aliases = meta.get('array_aliases', {})
        self.integer_dtypes = meta.get('integer_dtypes', {})
        if meta['version'] >= 6 and not {'array_aliases', 'integer_dtypes'} <= meta.keys():
            raise ValueError('Descritores de armazenamento ausentes no projeto.')
        if not isinstance(self.aliases, dict) or not isinstance(self.integer_dtypes, dict):
            raise ValueError('Descritores de armazenamento inválidos no projeto.')
        allowed = {'centers', 'center', 'scale', 'palette'}
        integer_keys = set()
        for i, info in enumerate(meta['items']):
            if type(info['image_count']) is not int or info['image_count'] < 1:
                raise ValueError('Metadados de imagem inválidos no projeto.')
            allowed.update(f'{i}_{key}' for key in (
                'rgb', 'initial', 'labels', 'valid', 'locked', 'excluded', 'applied_excluded',
                'confidence', 'sample_X', *_OPTIONAL_OPTICAL_ARRAYS))
            integer_keys.update(f'{i}_{key}' for key in ('initial', 'labels', 'locked'))
            allowed.update(f'{i}_image_{j}' for j in range(info['image_count']))
            allowed.update(f'{i}_registered_overlap_{j}' for j in range(info['image_count']))
            for c in info.get('annotations', []):
                allowed.update((f'{i}_annotation_{c}', f'{i}_applied_annotation_{c}'))
            allowed.update(f'{i}_background_mask_{c}' for c in info.get('background_masks', []))
        physical = set(archive.files)
        if len(physical) != len(archive.files):
            raise ValueError('Arrays duplicados no projeto.')
        for key, target in self.aliases.items():
            if (key not in allowed or not isinstance(target, str) or target not in allowed
                    or key in physical or target in self.aliases):
                raise ValueError('Imagem ou máscara com referência de array inválida no projeto.')
            # Older schema fixtures can intentionally omit applied-state members.
            if meta['version'] >= 6 and target not in physical:
                raise ValueError('Referência de array ausente no projeto.')
        for key, dtype in self.integer_dtypes.items():
            if (key not in integer_keys or not isinstance(dtype, str)
                    or dtype not in ('<i4', '>i4', '<i8', '>i8', '<u4', '>u4', '<u8', '>u8')
                    or (meta['version'] >= 6 and key not in physical and key not in self.aliases)):
                raise ValueError('Rótulos com representação inteira inválida no projeto.')
        self.files = list(dict.fromkeys([*archive.files, *self.aliases]))

    def __contains__(self, key):
        return key in self.files

    def __getitem__(self, key):
        array = self.archive[self.aliases.get(key, key)]
        if key in self.integer_dtypes:
            dtype = np.dtype(self.integer_dtypes[key])
            if (array.dtype != np.int16 or np.any((array < -1) | (array > 254))
                    or (dtype.kind == 'u' and np.any(array < 0))):
                raise ValueError('Rótulos com representação inteira inválida no projeto.')
            array = array.astype(dtype)
        return array


@export_operation
def save_project(prepared, path):
    """Atomically store a self-contained numeric archive; never use pickle."""
    state = prepared['review']
    arrays = {}
    metadata = dict(version=6, mode=state.mode, config=asdict(state.config), dirty=sorted(state.dirty),
                    sample_layout='shared' if state.mode == 'independent' else 'per_item',
                    output_history=_project_output_history(prepared),
                    exclusion_dirty=state.exclusion_dirty,
                    unresolved_dirty=sorted(state.unresolved_dirty),
                    background_groups=sorted(state.background_groups),
                    applied_background_groups=sorted(state.applied_background_groups),
                    groups=[sorted(state.groups[c]) for c in state.class_ids],
                    completed=sorted(state.completed), selected_image=state.selected_image,
                    selected_cluster=state.selected_cluster, output=prepared['output'],
                    workspace=prepared.get('workspace', {}),
                    interpretation=prepared['interpretation'].to_dict(), items=[],
                    models={str(c): _model_text(m) for c, m in state.models.items()},
                    refinement_order=list(state.refinement_order))
    if state._catalog_edited:
        metadata.update(version=7,
                        group_catalog=[dict(id=c, members=sorted(state.groups[c])) for c in state.class_ids],
                        group_names={str(c): name for c, name in state.group_names.items()},
                        retired_groups={str(c): entry for c, entry in state.retired_groups.items()},
                        next_group_id=state.next_group_id,
                        legacy_group_names={str(c): name for c, name in state._legacy_group_names.items()})
    for key in ('centers', 'center', 'scale', 'palette'):
        arrays[key] = getattr(state, key)
    for i, item in enumerate(state.items):
        raw = item['optical']
        info = dict(id=i, name=item['name'], path=item['path'],
                    image_names=raw['image_names'], reference_index=raw['reference_index'],
                    mask_mode=raw.get('mask_mode', 'auto'), annotations=sorted(item['annotations']),
                    background_masks=sorted(item['background_masks']),
                    image_count=len(raw['images']), overlap_status=raw.get('overlap_status'),
                    optional_arrays=[key for key in _OPTIONAL_OPTICAL_ARRAYS if key in raw],
                    registered_images=[{key: value for key, value in entry.items() if key not in ('rgb', 'overlap')}
                                       for entry in raw.get('registered_images', [])])
        metadata['items'].append(info)
        for key in info['optional_arrays']:
            arrays[f'{i}_{key}'] = raw[key]
        for j, entry in enumerate(raw.get('registered_images', [])):
            if 'overlap' in entry:
                arrays[f'{i}_registered_overlap_{j}'] = entry['overlap']
        for key in ('rgb', 'initial', 'labels', 'valid', 'locked', 'excluded', 'applied_excluded'):
            arrays[f'{i}_{key}'] = item[key]
        arrays[f'{i}_confidence'] = raw['confidence']
        if state.mode != 'independent' or i == 0:
            arrays[f'{i}_sample_X'] = raw['sample_X']
        elif (raw['sample_X'] is not state.items[0]['optical']['sample_X']
              and (raw['sample_X'].dtype != state.items[0]['optical']['sample_X'].dtype
                   or not np.array_equal(raw['sample_X'], state.items[0]['optical']['sample_X']))):
            # Historical projects may contain legitimate distinct samples. Keep
            # those as per-item data rather than discarding training evidence.
            metadata['sample_layout'] = 'per_item'
    if metadata['sample_layout'] == 'per_item':
        for i, item in enumerate(state.items):
            arrays[f'{i}_sample_X'] = item['optical']['sample_X']
    for i, item in enumerate(state.items):
        raw = item['optical']
        for c, marks in item['annotations'].items():
            arrays[f'{i}_annotation_{c}'] = marks
            arrays[f'{i}_applied_annotation_{c}'] = item['applied_annotations'].get(c, np.zeros_like(marks))
        for c, mask in item['background_masks'].items():
            arrays[f'{i}_background_mask_{c}'] = mask
        for j, image in enumerate(raw['images']):
            arrays[f'{i}_image_{j}'] = image
    arrays, metadata['array_aliases'], metadata['integer_dtypes'] = _compact_project_arrays(arrays)
    arrays['metadata'] = np.array(json.dumps(metadata, ensure_ascii=False, default=_json_value))
    path = Path(path)
    path = staged_file(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def _restore_group_catalog(state, meta):
    """Validate stable identities independently of the historical raw partition."""
    catalog, next_id = meta['group_catalog'], meta['next_group_id']
    if (not isinstance(catalog, list) or not catalog or type(next_id) is not int
            or next_id < state.k or next_id > np.iinfo(np.int32).max - state.config.seed):
        raise ValueError('Catálogo de grupos inválido.')
    groups, seen = {}, set()
    for entry in catalog:
        if not isinstance(entry, dict) or set(entry) != {'id', 'members'}:
            raise ValueError('Identidade de grupo inválida.')
        group, members = entry['id'], entry['members']
        if (type(group) is not int or not 0 <= group < next_id or group in groups
                or not isinstance(members, list) or any(type(c) is not int or not 0 <= c < state.k for c in members)
                or len(members) != len(set(members)) or seen.intersection(members)):
            raise ValueError('Associações do catálogo de grupos inválidas.')
        groups[group] = set(members)
        seen.update(members)
    def names(field):
        incoming = meta[field]
        if not isinstance(incoming, dict):
            raise ValueError('Nomes do catálogo de grupos inválidos.')
        result = {}
        for key, name in incoming.items():
            if (not isinstance(key, str) or not key.isdecimal() or str(int(key)) != key
                    or int(key) not in groups or not isinstance(name, str)):
                raise ValueError('Nomes do catálogo de grupos inválidos.')
            result[int(key)] = name
        return result
    retired = meta['retired_groups']
    if not isinstance(retired, dict):
        raise ValueError('Proveniência de grupos excluídos inválida.')
    retired_groups = {}
    for key, entry in retired.items():
        if (not isinstance(key, str) or not key.isdecimal() or str(int(key)) != key
                or not 0 <= int(key) < next_id or int(key) in groups or not isinstance(entry, dict)
                or set(entry) != {'members', 'name', 'color', 'reason'}
                or not isinstance(entry['members'], list)
                or any(type(c) is not int or not 0 <= c < state.k for c in entry['members'])
                or len(entry['members']) != len(set(entry['members']))
                or not isinstance(entry['name'], str) or not isinstance(entry['reason'], str)
                or not isinstance(entry['color'], list) or len(entry['color']) != 3
                or any(type(v) is not int or not 0 <= v <= 255 for v in entry['color'])):
            raise ValueError('Proveniência de grupos excluídos inválida.')
        retired_groups[int(key)] = deepcopy(entry)
    if seen | {c for entry in retired_groups.values() for c in entry['members']} != set(range(state.k)):
        raise ValueError('Proveniência dos clusters automáticos incompleta.')
    state.group_names = names('group_names')
    state._legacy_group_names = names('legacy_group_names')
    state.groups, state.retired_groups = groups, retired_groups
    state.next_group_id, state._catalog_edited = next_id, True


def load_project(path):
    """Restore exactly the saved masks, including unfinished annotations."""
    try:
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(str(archive['metadata']))
            if not isinstance(meta, dict):
                raise ValueError('Metadados de projeto inválidos.')
            if meta.get('version') not in (1, 2, 3, 4, 5, 6, 7) or meta.get('mode') not in ('independent', 'registered'):
                raise ValueError('Versão ou modo de projeto incompatível.')
            cfg = core.Config(**meta['config'])
            _validate_project_config(cfg)
            if (not isinstance(meta['items'], list) or not meta['items']
                    or (meta['mode'] == 'registered' and len(meta['items']) != 1)):
                raise ValueError('Quantidade de imagens inválida no projeto.')
            archive = _ProjectArrays(archive, meta)
            output_history = _project_output_history(meta)
            shared_sample = False
            if meta['version'] >= 5:
                if 'output_history' not in meta:
                    raise ValueError('Histórico de destinos ausente no projeto.')
                layout = meta.get('sample_layout')
                if layout not in ('shared', 'per_item') or (layout == 'shared' and meta['mode'] != 'independent'):
                    raise ValueError('Formato da amostra inválido no projeto.')
                shared_sample = layout == 'shared'
                expected = {'0_sample_X'} if shared_sample else {f'{i}_sample_X' for i in range(len(meta['items']))}
                if {key for key in archive.files if key.endswith('_sample_X')} != expected:
                    raise ValueError('Arrays da amostra incompatíveis com o formato do projeto.')
            palette = archive['palette'].copy()
            palette_count = meta.get('next_group_id') if meta['version'] >= 7 else cfg.clusters
            if (type(palette_count) is not int or palette_count < cfg.clusters
                    or palette.dtype != np.uint8 or palette.shape != (palette_count, 3)):
                raise ValueError('Paleta de cores inválida no projeto.')
            items = []
            for i, info in enumerate(meta['items']):
                image_count = info['image_count']
                if (type(image_count) is not int or image_count < 1
                        or (meta['mode'] == 'independent' and image_count != 1)
                        or not isinstance(info['image_names'], list)
                        or len(info['image_names']) != image_count
                        or any(not isinstance(name, str) for name in info['image_names'])
                        or type(info['reference_index']) is not int
                        or not 0 <= info['reference_index'] < image_count):
                    raise ValueError('Metadados de imagem inválidos no projeto.')
                if info['mask_mode'] not in ('auto', 'overlap_only'):
                    raise ValueError('Modo de máscara inválido no projeto.')
                rgb, valid = archive[f'{i}_rgb'].copy(), archive[f'{i}_valid'].copy()
                initial, confidence = archive[f'{i}_initial'].copy(), archive[f'{i}_confidence'].copy()
                if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1:
                    raise ValueError('Imagem RGB inválida no projeto.')
                if initial.shape != rgb.shape[:2] or valid.shape != initial.shape:
                    raise ValueError('Geometria inválida no projeto.')
                if valid.dtype != bool or not np.any(valid):
                    raise ValueError('Máscara de análise inválida no projeto.')
                if (confidence.dtype.kind not in 'fiu' or confidence.shape != initial.shape
                        or not np.all(np.isfinite(confidence)) or np.any((confidence < 0) | (confidence > 1))):
                    raise ValueError('Confiança óptica inválida no projeto.')
                if initial.dtype.kind not in 'iu' or np.any((initial < 0) | (initial >= cfg.clusters)):
                    raise ValueError('Rótulos iniciais inválidos no projeto.')
                images = [archive[f'{i}_image_{j}'].copy() for j in range(image_count)]
                if any(image.dtype != np.uint8 or image.shape != rgb.shape for image in images):
                    raise ValueError('Imagem de análise inválida ou com geometria incompatível.')
                dimensions = 4 * image_count
                centers, center, scale = (archive[key].copy() for key in ('centers', 'center', 'scale'))
                for array, shape in ((centers, (cfg.clusters, dimensions)), (center, (dimensions,)),
                                     (scale, (dimensions,))):
                    _validate_feature_array(array, shape)
                if np.any(scale <= 0) or np.any(scale.astype(np.float32) <= 0):
                    raise ValueError('Escala de atributos inválida no projeto.')
                sample = (items[0]['optical']['sample_X'] if shared_sample and i else
                          archive[f'{i}_sample_X'].copy())
                _validate_feature_array(sample, (None, dimensions))
                if len(sample) < cfg.clusters:
                    raise ValueError('Amostra de atributos insuficiente no projeto.')
                if meta['mode'] == 'independent' and not shared_sample:
                    for previous in items:
                        other = previous['optical']['sample_X']
                        if other.dtype == sample.dtype and np.array_equal(other, sample):
                            sample = other
                            break
                optional = _load_optional_optical_arrays(archive, i, info, valid, cfg.clusters, dimensions)
                optical = dict(reference=rgb, images=images, valid=valid, labels=initial,
                               confidence=confidence, sample_X=sample,
                               clusters=cfg.clusters, centers=centers, center=center, scale=scale,
                               palette=optional.pop('palette', palette[:cfg.clusters].copy()), image_names=info['image_names'],
                               reference_index=info['reference_index'], mask_mode=info['mask_mode'],
                               registered_images=[dict(name=n) for n in info['image_names']],
                               feature_blur_sigma=cfg.feature_blur_sigma,
                               stats=_statistics(initial, valid, confidence, cfg.clusters))
                optical.update(optional)
                optical['overlap_status'] = info.get('overlap_status')
                if info.get('registered_images'):
                    if len(info['registered_images']) != image_count:
                        raise ValueError('Metadados de imagem registrada incompletos.')
                    registered = []
                    for j, entry in enumerate(info['registered_images']):
                        entry = dict(entry, rgb=images[j])
                        overlap_key = f'{i}_registered_overlap_{j}'
                        if overlap_key in archive:
                            entry['overlap'] = archive[overlap_key].copy()
                        if 'registration' in entry:
                            reg = entry['registration']
                            reg['matrix'] = np.asarray(reg['matrix'], np.float32)
                            for field in ('translation', 'shift_dx_dy'):
                                if field in reg:
                                    reg[field] = tuple(reg[field])
                        registered.append(entry)
                    optical['registered_images'] = registered
                initial.setflags(write=False)
                items.append(dict(id=i, name=info['name'], path=info['path'], rgb=rgb, valid=valid,
                                  optical=optical, sources=None, initial=initial,
                                  labels=archive[f'{i}_labels'].copy(),
                                  locked=archive[f'{i}_locked'].copy(),
                                  annotations={c: archive[f'{i}_annotation_{c}'].copy() for c in info['annotations']}))
                item = items[-1]
                for key in ('excluded', 'applied_excluded'):
                    item[key] = (archive[f'{i}_{key}'].copy() if meta['version'] >= 3
                                 else np.zeros(initial.shape, bool))
                item['applied_annotations'] = {
                    c: (archive[f'{i}_applied_annotation_{c}'].copy() if meta['version'] >= 3 else
                        np.zeros_like(marks) if c in meta.get('dirty', []) else marks.copy())
                    for c, marks in item['annotations'].items()}
                item['display_valid'] = item['valid'] & ~item['applied_excluded']
                item['background_masks'] = {c: archive[f'{i}_background_mask_{c}'].copy()
                                            for c in info.get('background_masks', [])}
            if not items:
                raise ValueError('Projeto sem imagens.')
            prepared = dict(output=_portable_output(path, output_history), output_history=output_history,
                            output_pending=True,
                            config=cfg, optical=items[0]['optical'],
                            analysis_mode=meta['mode'], dataset_items=items,
                            workspace=meta.get('workspace', {}),
                            review_groups=(meta['interpretation']['groups'] if meta['version'] >= 7 else
                                           meta.get('groups', core.identity_groups(cfg.clusters))))
            prepared['interpretation'] = core.Interpretation(prepared['optical'])
            prepared['interpretation']._restore(meta['interpretation'])
            state = ClusterReview.from_prepared(prepared)
            state.items = items
            state.palette = archive['palette'].copy()
            if meta['version'] >= 7:
                _restore_group_catalog(state, meta)
            state.completed = set(meta['completed'])
            state.dirty = set(meta.get('dirty', []))
            state.exclusion_dirty = bool(meta.get('exclusion_dirty', False))
            state.background_groups = set(meta.get('background_groups', []))
            state.applied_background_groups = set(meta.get('applied_background_groups', []))
            state.group_policy_dirty = state.background_groups != state.applied_background_groups
            if ((state.background_groups | state.applied_background_groups) - set(state.class_ids)
                    or (state.background_groups | state.applied_background_groups) & state.completed):
                raise ValueError('Grupos de fundo inválidos.')
            state.unresolved_dirty = (set(state.dirty) if meta['version'] < 3
                                      else set(meta.get('unresolved_dirty', [])))
            if not state.unresolved_dirty <= state.dirty:
                raise ValueError('Pendências de revisão inválidas.')
            if not state.completed <= set(state.class_ids) or not state.dirty <= set(state.class_ids):
                raise ValueError('Cluster aprovado inválido.')
            if state.completed & state.dirty:
                raise ValueError('Pendências de grupo aprovado inválidas.')
            for item in items:
                for key in ('labels', 'locked'):
                    arr = item[key]
                    if (arr.dtype.kind not in 'iu' or arr.shape != item['valid'].shape
                            or np.any((arr < -1) | (arr >= state.next_group_id))):
                        raise ValueError('Rótulos inválidos no projeto.')
                    item[key] = arr.astype(np.int32)
                if np.any(item['valid'] & ~np.isin(item['labels'], [-1] + state.class_ids)):
                    raise ValueError('Rótulos de grupo inválidos no projeto.')
                frozen = item['locked'] >= 0
                if np.any(~item['valid'] & frozen) or not set(np.unique(item['locked'][frozen])) <= state.completed:
                    raise ValueError('Máscara aprovada inválida.')
                if np.any(item['labels'][frozen] != item['locked'][frozen]):
                    raise ValueError('Pixels aprovados foram alterados no arquivo.')
                if np.any(item['valid'] & np.isin(item['labels'], list(state.completed)) & ~frozen):
                    raise ValueError('Pixels aprovados estão sem proteção no arquivo.')
                for key in ('excluded', 'applied_excluded'):
                    mask = item[key]
                    if mask.dtype != bool or mask.shape != item['valid'].shape or np.any(mask & (~item['valid'] | frozen)):
                        raise ValueError('Máscara de fundo inválida.')
                if np.any(item['labels'][item['applied_excluded']] != -1):
                    raise ValueError('Pixels de fundo foram reclassificados no arquivo.')
                if (not state.exclusion_dirty
                        and not np.array_equal(item['excluded'], item['applied_excluded'])):
                    raise ValueError('Pendências de fundo inválidas.')
                if set(item['background_masks']) != state.background_groups:
                    raise ValueError('Máscaras de recuperação de grupos de fundo incompletas.')
                for c, mask in item['background_masks'].items():
                    # This is historical provenance. Individually restored pixels
                    # can subsequently belong to another approved group. exclude()
                    # enforces the locks when restoring the full group later.
                    if (c not in state.background_groups or mask.dtype != bool or mask.shape != item['valid'].shape
                            or np.any(mask & ~item['valid'])):
                        raise ValueError('Máscara de grupo de fundo inválida.')
                for c, marks in item['annotations'].items():
                    if c not in state.groups or marks.shape != item['valid'].shape or np.any(~np.isin(marks, [-1, 0, 1])):
                        raise ValueError('Anotações inválidas.')
                    applied = item['applied_annotations'][c]
                    if applied.shape != marks.shape or np.any(~np.isin(applied, [-1, 0, 1])):
                        raise ValueError('Anotações aplicadas inválidas.')
                    if c not in state.dirty and not np.array_equal(marks, applied):
                        raise ValueError('Pendências de anotações inválidas.')
            state.models = {}
            for c, text in meta['models'].items():
                ml = supervised_backend()
                storage = cv2.FileStorage(text, cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY)
                model = ml.RTrees_create()
                model.read(storage.getNode('model'))
                storage.release()
                if not model.isTrained() or model.getVarCount() != len(state.center):
                    raise ValueError('Modelo de aprendizado inválido.')
                if int(c) not in state.groups:
                    raise ValueError('Grupo do modelo inválido.')
                state.models[int(c)] = model
            if 'refinement_order' in meta:
                order = meta['refinement_order']
                if (not isinstance(order, list) or any(type(c) is not int or c not in state.groups for c in order)
                        or len(set(order)) != len(order)):
                    raise ValueError('Ordem de precedência do refinamento inválida.')
                state.refinement_order = list(order)
            else:
                # Restore historical precedence without computing features or
                # changing the saved classification during opening.
                one_sign = []
                for c in state.class_ids:
                    signs = set()
                    for item in items:
                        marks = item['annotations'].get(c)
                        if marks is not None:
                            allowed = item['valid'] & ~item['excluded'] & (item['locked'] < 0)
                            signs.update(int(value) for value in np.unique(marks[allowed]) if value)
                    if len(signs) == 1 and c not in state.models:
                        one_sign.append(c)
                state.refinement_order = one_sign + list(state.models)
            state.selected_image, state.selected_cluster = int(meta['selected_image']), int(meta['selected_cluster'])
            if state.selected_cluster not in state.groups:
                raise ValueError('Seleção de cluster inválida.')
            limit = len(items) if state.mode == 'independent' else len(items[0]['optical']['images'])
            if not 0 <= state.selected_image < limit:
                raise ValueError('Seleção de imagem inválida.')
            prepared['review'] = state
            prepared['workspace'] = _validate_workspace(prepared['workspace'], state.class_ids)
            return prepared
    except (KeyError, TypeError, ValueError, OSError, EOFError, zipfile.BadZipFile, cv2.error) as exc:
        raise ValueError(f'Projeto RIIMA inválido: {exc}') from exc


@export_operation
def save_review_outputs(prepared):
    """Export review products per image, keeping the original outputs separate."""
    state = prepared['review']
    out = staged_directory(Path(prepared['output']) / 'review')
    out.mkdir(parents=True, exist_ok=True)
    catalog = dict(next_group_id=state.next_group_id,
                   groups=[dict(id=c, source_clusters=sorted(state.groups[c]),
                                name=state.group_name(c, prepared.get('interpretation')),
                                color=state.palette[c].tolist(), approved=c in state.completed,
                                background=c in state.background_groups, pixels=state.counts(c))
                           for c in state.class_ids],
                   retired_groups={str(c): entry for c, entry in state.retired_groups.items()})
    (out / 'catalog.json').write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding='utf-8')
    with (out / 'statistics.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['image', 'group', 'source_clusters', 'pixels', 'approved'])
        for i, item in enumerate(state.items):
            folder = out / f'image_{i:03d}'
            folder.mkdir(exist_ok=True)
            np.save(folder / 'labels.npy', item['labels'])
            np.save(folder / 'approved.npy', item['locked'])
            np.save(folder / 'excluded.npy', item['applied_excluded'])
            rgb = np.full_like(item['rgb'], 30)
            assigned = item['valid'] & (item['labels'] >= 0)
            rgb[assigned] = state.palette[item['labels'][assigned]]
            core._save_png_fast(rgb, folder / 'clusters.png')
            core._save_png_fast(core.render_optical_clusters(item['initial'], item['valid'], item['optical']['palette']), folder / 'initial.png')
            for c in state.class_ids:
                writer.writerow([item['name'], c, json.dumps(sorted(state.groups[c])),
                                 int((item['valid'] & (item['labels'] == c)).sum()), c in state.completed])
