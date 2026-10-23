"""Pure contrast and scientific raster contracts; real Tk tests share test_riima_gui."""
import hashlib
import json

import numpy as np
from riima import theme, viewer


def _contrast(a, b):
    def luminance(color):
        channels = [int(color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        linear = [c / 12.92 if c <= .04045 else ((c + .055) / 1.055) ** 2.4 for c in channels]
        return sum(c * weight for c, weight in zip(linear, (.2126, .7152, .0722)))
    light, dark = sorted((luminance(a), luminance(b)), reverse=True)
    return (light + .05) / (dark + .05)


def test_body_metadata_and_accent_text_meet_contrast():
    assert _contrast(theme.TEXT_SECONDARY, theme.BG_APP) >= 4.5
    assert _contrast(theme.TEXT_ON_ACCENT, theme.ACCENT) >= 4.5
    assert _contrast(theme.WARNING, theme.BG_APP) >= 4.5


def test_scientific_palette_masks_and_annotation_pixels_are_preserved(monkeypatch):
    base = np.arange(18, dtype=np.uint8).reshape(2, 3, 3) * 10
    palette = np.array([(17, 83, 149), (201, 32, 167)], dtype=np.uint8)
    current = dict(base=base, labels=np.array([[0, 1, 0], [1, -1, -1]]),
                   valid=np.array([[True, True, True], [True, True, False]]),
                   palette=palette, annotations={0: np.array([[1, -1, 0], [0, 0, 0]])},
                   applied_annotations={0: np.array([[0, 0, 1], [0, 0, 0]])})
    monkeypatch.setattr(viewer, 'workspace_data', lambda prepared: current)
    classified = viewer.render_workspace({}, 'Classificação')
    np.testing.assert_array_equal(classified, np.array([[palette[0], palette[1], palette[0]],
                                                       [palette[1], (30, 30, 30), (30, 30, 30)]], dtype=np.uint8))
    mask = viewer.render_workspace({}, 'Máscara selecionada', selected_groups=[0])
    np.testing.assert_array_equal(mask[:, :, 0], [[255, 0, 255], [0, 0, 0]])
    marked = viewer.render_workspace({}, 'Imagem', annotation_group=0)
    np.testing.assert_array_equal(marked[0], [(0, 255, 100), (255, 70, 70), (70, 210, 255)])
    np.testing.assert_array_equal(marked[1], base[1])


def _scientific_viewer_fixture():
    """Known pixel grids and metadata: no clustering, RNG, image files or cache."""
    from riima import core, review

    labels = np.array([[0, 0, 0, 0, 1, 1, 1, 1],
                       [0, 0, 0, 0, 1, 1, 1, 1],
                       [0, 0, 0, 0, 1, 1, 1, 1],
                       [0, 0, 0, 0, 1, 1, 1, 1],
                       [2, 2, 2, 2, 1, 1, 1, 1],
                       [2, 2, 2, 2, 1, 1, 1, 1]], dtype=np.uint8)
    rgb = np.arange(144, dtype=np.uint8).reshape(6, 8, 3)
    valid = np.ones((6, 8), dtype=bool)
    valid[0, 0] = valid[5, 7] = False
    confidence = np.tile(np.array([0, .25, .5, .75, 1, .75, .5, .25], dtype=np.float32), (6, 1))
    palette = np.array([(17, 83, 149), (201, 32, 167), (119, 183, 41)], dtype=np.uint8)
    optical = dict(labels=labels, reference=rgb, images=[rgb], valid=valid,
                   confidence=confidence, clusters=3, palette=palette,
                   centers=np.arange(12, dtype=np.float32).reshape(3, 4),
                   center=np.zeros(4, dtype=np.float32), scale=np.ones(4, dtype=np.float32),
                   stats=[dict(cluster=c, area=area, confidence=score)
                          for c, area, score in [(0, 32.6, .25), (1, 50., .75), (2, 17.4, .5)]])
    interpretation = core.Interpretation(optical)
    interpretation.rename_cluster(0, 'Quartzo alterado: nome completo')
    interpretation.recolor_cluster(0, (23, 91, 157))
    interpretation.create_family([0, 2], name='Associação α', color=(93, 101, 117))
    prepared = dict(optical=optical, interpretation=interpretation, config=core.Config(clusters=3))
    state = review.ClusterReview.from_prepared(prepared)
    prepared['review'] = state
    # Edited workspace color is deliberately distinct from raw/interpreted colors.
    state.palette[0] = (29, 97, 163)
    state.completed = {1}
    state.dirty = {0}
    item = state.items[0]
    item['annotations'][0] = np.zeros((6, 8), dtype=np.int8)
    item['applied_annotations'][0] = np.zeros((6, 8), dtype=np.int8)
    item['annotations'][0][1, 1] = 1
    item['annotations'][0][1, 2] = -1
    item['applied_annotations'][0][2, 1] = 1  # erased pending annotation
    item['excluded'][4, 4] = True
    item['applied_excluded'][4, 5] = True  # restored pending area
    item['display_valid'][4, 4] = False
    item['labels'][4, 4] = -1
    return prepared


def _rgb_signature(array):
    return (tuple(array.shape), str(array.dtype), hashlib.sha256(array.tobytes(order='C')).hexdigest())


def _legend_signature(rows):
    encoded = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


_WORKSPACE_DISPLAYS = ('Imagem', 'Máscara selecionada', 'Classificação', 'Sobreposição', 'Grupo selecionado')
_PREVIEW_MODES = ('Clusters em revisão', 'Clusters brutos', 'Clusters interpretados', 'Confiança óptica',
                  'Famílias', 'Máscara', 'Contornos de clusters', 'Contornos de famílias')


# Frozen from the original viewer, before visual centralization, using the fixture
# above. Hash input includes exact RGB bytes; shape and dtype are asserted as well.
# Legend hashes use UTF-8 canonical JSON and preserve row order and full metadata.
# The review legend follows current interpretive names; its label-only update
# leaves every raster snapshot, color, count and status unchanged.
_EXPECTED_WORKSPACE = {('Imagem', False): ((6, 8, 3), 'uint8', '63b333a234ad1f2ccfd4bcee4e3aa5a145e05b48e7c7c2f1bc2ecb906ee05935'),
 ('Imagem', True): ((6, 8, 3), 'uint8', '4bdc96ddbe8c1324667e1a01e4eb23c5a6057d2870f22dad2d18270b44d6140a'),
 ('Máscara selecionada', False): ((6, 8, 3),
                                  'uint8',
                                  '39683e74054308e0ecedc2819126113ce4166670cd2fd94abb4664af4a4ee888'),
 ('Máscara selecionada', True): ((6, 8, 3),
                                 'uint8',
                                 '39683e74054308e0ecedc2819126113ce4166670cd2fd94abb4664af4a4ee888'),
 ('Classificação', False): ((6, 8, 3), 'uint8', 'bbb5066248f37c5f777f66c0a867a66dbf1f6ba442c24412a98a764d426d1ff4'),
 ('Classificação', True): ((6, 8, 3), 'uint8', '2fef2c3c1ad7c1d78c95005084940f3f48764813ac54b849965feca630e997c1'),
 ('Sobreposição', False): ((6, 8, 3), 'uint8', 'ac7264b8c6d299a3b25895afb7eee405ef7182d141e85ec1b44fb429d8eacbb8'),
 ('Sobreposição', True): ((6, 8, 3), 'uint8', '95f482aae5104df8ec6c084e9924ce921688566c0351dd699f96317c3dff4ebc'),
 ('Grupo selecionado', False): ((6, 8, 3), 'uint8', 'ac7264b8c6d299a3b25895afb7eee405ef7182d141e85ec1b44fb429d8eacbb8'),
 ('Grupo selecionado', True): ((6, 8, 3), 'uint8', '95f482aae5104df8ec6c084e9924ce921688566c0351dd699f96317c3dff4ebc')}
_EXPECTED_PREVIEW = {'Clusters em revisão': ((6, 8, 3), 'uint8', '0a881d455545615ea3df40403d9c0103fd3b6ad1f66abac8df3bb27c470887cd'),
 'Clusters brutos': ((6, 8, 3), 'uint8', '2e367745838d0118dd9f0a005b4f3f4171860ca294c4f5326c52f8aa15c0438b'),
 'Clusters interpretados': ((6, 8, 3), 'uint8', 'f6e56dd9a6277a5296dc792b6a9db7d3bc7132d0becd941aded870b9bb8b93bd'),
 'Confiança óptica': ((6, 8, 3), 'uint8', 'a2528da863e79625267b2dcfed1b0c5abae88de5c0452cb0329a5d466d5c4ba0'),
 'Famílias': ((6, 8, 3), 'uint8', 'c6b1ff0d6b612938ff3ebd6204ce23e2f22f4fa0290f9afc2b863039c7ab91af'),
 'Máscara': ((6, 8, 3), 'uint8', '82c9f55bce8e82c73fa898b7b744ff40d4c55f6d6421679ecd74a03e6dc7070a'),
 'Contornos de clusters': ((6, 8, 3), 'uint8', '2c791d5d86270bbed7c9833d9af5fb8ac52bcf903a62b204e8b0a89bcc4a8d5e'),
 'Contornos de famílias': ((6, 8, 3), 'uint8', '73ff4abb74644071528bb8d4c8694d304e8dfaff5c73fc0ed679b65de5ff0050')}
_EXPECTED_LEGENDS = {'workspace': '7c82cc51d2eeeed040661ac4b18f1571aff2ad3922761985ea30bf0fdbffa0fc',
 'Clusters em revisão': '683e296ceeea0df309d3e88bb81f306aa9a834ed9174e0450b8fd86cc6261f26',
 'Clusters brutos': 'f48d0a776362c66d82deacf2e66bd6fa5154475c610e53047929f689c4332fd2',
 'Clusters interpretados': '4fe7d6746051447cb38ab9d990d37edce27c3da2017cc8d9355303414a4b0a3a',
 'Confiança óptica': '15cdb91946b92d2a12106f830c2196b21c63c95b013d15e252bd4afe3951cde8',
 'Famílias': 'eeaf5c3f512c4bec222c0c7033230773a5bdc26d52d6d23a182f9bd6de7975f2',
 'Máscara': 'f0d440fb841b9295aa643c78191ffefbafea5ef8dcaa75d278b66411208bca8d',
 'Contornos de clusters': '8b3ffad0987b9aef4920524d4824295f41281b201717966940c1c92fa96be168',
 'Contornos de famílias': 'fab9fba3fafda0409cefa19c1d66fa8dcde13d89c23822a70a794f42a7195223'}


def test_viewer_scientific_pixels_match_original_snapshot():
    prepared = _scientific_viewer_fixture()
    assert [row['label'] for row in viewer.build_preview_legend('Clusters em revisão', prepared)] == [
        'Quartzo alterado: nome completo', 'Grupo 1', 'Grupo 2']
    for display in _WORKSPACE_DISPLAYS:
        for contours in (False, True):
            rgb = viewer.render_workspace(prepared, display, selected_groups=[0], opacity=.37,
                                          contours=contours, annotation_group=0)
            assert _rgb_signature(rgb) == _EXPECTED_WORKSPACE[(display, contours)]
    assert _legend_signature(viewer.workspace_legend(prepared)) == _EXPECTED_LEGENDS['workspace']
    for mode in _PREVIEW_MODES:
        assert _rgb_signature(viewer.render_preview_mode(mode, prepared)) == _EXPECTED_PREVIEW[mode]
        assert _legend_signature(viewer.build_preview_legend(mode, prepared)) == _EXPECTED_LEGENDS[mode]
