"""Dataset-wide interactive cluster refinement and resumable projects.

Initial optical labels are immutable. Review labels and approved masks have stable
group IDs derived from the raw cluster IDs. This module contains no Tk code and
never modifies input images.
"""
from __future__ import annotations

from dataclasses import asdict
import csv
import json
import os
from pathlib import Path
import tempfile
import zipfile

import cv2
import numpy as np

from . import core
from .exports import export_operation, staged_directory, staged_file
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
        state.centers = np.asarray(optical['centers'], np.float32).copy()
        state.center = np.asarray(optical['center'], np.float32).copy()
        state.scale = np.asarray(optical['scale'], np.float32).copy()
        state.palette = np.asarray(optical['palette'], np.uint8).copy()
        state.completed, state.models, state.items, state.dirty = set(), {}, [], set()
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
        return state

    @property
    def class_ids(self):
        return sorted(self.groups)

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
        lut = np.empty(self.k, np.int32)
        for group, members in self.groups.items():
            lut[list(members)] = group
        return lut[item['initial']]

    def sync_groups(self, groups):
        """Adopt the interpretation partition without destroying existing reviews."""
        partition = core._validate_partition([set(g) for g in groups], self.k)
        incoming = {min(g): set(g) for g in partition}
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

    def _fit_model(self, cluster):
        """Fit one group from accumulated examples, without performing prediction."""
        self._cluster(cluster, editable=True)
        examples, responses = [], []
        rng = np.random.default_rng(self.config.seed + cluster)
        for item in self.items:
            marks = item['annotations'].get(cluster)
            if marks is None:
                continue
            for value in (-1, 1):
                idx = np.flatnonzero((marks == value) & item['valid'] & ~item['excluded'] & (item['locked'] < 0))
                if len(idx) > 5000:
                    idx = rng.choice(idx, 5000, replace=False)
                if len(idx):
                    examples.append(self._features(item, idx))
                    responses.append(np.full(len(idx), int(value == 1), np.int32))
        if responses and len(np.unique(np.concatenate(responses))) == 2:
            X, y = np.concatenate(examples), np.concatenate(responses)
            # Balance brush sizes, preserving every sampled explicit example.
            positive, negative = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
            n = max(len(positive), len(negative), 32)
            train_idx = np.concatenate([positive, rng.choice(positive, n - len(positive), replace=True),
                                        negative, rng.choice(negative, n - len(negative), replace=True)])
            cv2.setRNGSeed(self.config.seed + cluster)
            model = cv2.ml.RTrees_create()
            model.setMaxDepth(12)
            model.setMinSampleCount(1)
            model.setActiveVarCount(X.shape[1])
            model.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 64, 0))
            if not model.train(X[train_idx], cv2.ml.ROW_SAMPLE, y[train_idx]):
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
        try:
            trained = self._fit_model(cluster)
            self._predict(preferred=cluster)
        except Exception:
            self.models = previous_models
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
        model_order = list(previous_models)
        try:
            trained = {c: self._fit_model(c) for c in pending if c not in self.background_groups}
            if self.exclusion_dirty or self.group_policy_dirty:
                # Re-training for background alone must preserve learned precedence.
                self.models = {c: self.models[c] for c in dict.fromkeys(model_order + list(self.models))
                               if c in self.models}
            self._predict(preferred=preferred if preferred in edited else None)
        except Exception:
            self.models = previous_models
            raise
        self._commit_examples(pending)
        return trained

    def _predict(self, preferred=None):
        open_ids = [c for c in self.class_ids if c not in self.completed and c not in self.background_groups]
        # Latest edited model wins automatic conflicts; explicit positives always win.
        order = [c for c in self.models if c in open_ids and c != preferred]
        if preferred in self.models and preferred in open_ids:
            order.append(preferred)
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
                    for c in open_ids:
                        d = np.full(len(idx), np.inf, np.float32)
                        for raw_id in self.groups[c]:
                            d = np.minimum(d, np.sum((X - self.centers[raw_id]) ** 2, axis=1))
                        allowed = np.ones(len(idx), bool)
                        marks = item['annotations'].get(c)
                        if marks is not None:
                            allowed &= marks.ravel()[idx] != -1
                        model = self.models.get(c)
                        if model is not None:
                            _, predicted = model.predict(X)
                            predictions[c] = predicted.ravel() == 1
                            allowed &= predictions[c]
                        d = np.where(allowed, d, np.inf)
                        retained |= (previous == c) & allowed
                        take = d < best
                        best[take], labels[take] = d[take], c
                    # Keep the current spatially regularized selection where admissible.
                    labels[retained] = previous[retained]
                    for c in order:
                        take = predictions[c].copy()
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
    choices = dict(step=('images', 'classify', 'organize', 'refine', 'results'),
                   display=('Imagem', 'Sobreposição', 'Classificação', 'Máscara selecionada', 'Grupo selecionado'),
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


@export_operation
def save_project(prepared, path):
    """Atomically store a self-contained numeric archive; never use pickle."""
    state = prepared['review']
    arrays = {}
    metadata = dict(version=5, mode=state.mode, config=asdict(state.config), dirty=sorted(state.dirty),
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
                    models={str(c): _model_text(m) for c, m in state.models.items()})
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


def load_project(path):
    """Restore exactly the saved masks, including unfinished annotations."""
    try:
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(str(archive['metadata']))
            if not isinstance(meta, dict):
                raise ValueError('Metadados de projeto inválidos.')
            if meta.get('version') not in (1, 2, 3, 4, 5) or meta.get('mode') not in ('independent', 'registered'):
                raise ValueError('Versão ou modo de projeto incompatível.')
            cfg = core.Config(**meta['config'])
            _validate_project_config(cfg)
            if (not isinstance(meta['items'], list) or not meta['items']
                    or (meta['mode'] == 'registered' and len(meta['items']) != 1)):
                raise ValueError('Quantidade de imagens inválida no projeto.')
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
            if palette.dtype != np.uint8 or palette.shape != (cfg.clusters, 3):
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
                               palette=optional.pop('palette', palette.copy()), image_names=info['image_names'],
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
                            review_groups=meta.get('groups', core.identity_groups(cfg.clusters)))
            prepared['interpretation'] = core.Interpretation(prepared['optical'])
            prepared['interpretation']._restore(meta['interpretation'])
            state = ClusterReview.from_prepared(prepared)
            state.items = items
            state.palette = archive['palette'].copy()
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
                            or np.any((arr < -1) | (arr >= state.k))):
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
                storage = cv2.FileStorage(text, cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY)
                model = cv2.ml.RTrees_create()
                model.read(storage.getNode('model'))
                storage.release()
                if not model.isTrained() or model.getVarCount() != len(state.center):
                    raise ValueError('Modelo de aprendizado inválido.')
                if int(c) not in state.groups:
                    raise ValueError('Grupo do modelo inválido.')
                state.models[int(c)] = model
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
            core._save_png_fast(core.render_optical_clusters(state.initial_labels(item), item['valid'], state.palette), folder / 'initial.png')
            for c in state.class_ids:
                writer.writerow([item['name'], c, json.dumps(sorted(state.groups[c])),
                                 int((item['valid'] & (item['labels'] == c)).sum()), c in state.completed])
