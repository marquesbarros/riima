"""Scientific display semantics, independent of institutional interface colors.

Categorical colors belong to the current interpretation palette, indexed by ID.
The quantitative profiles retain their existing backend interpolation rules.
"""

UNASSIGNED_RGB = (30, 30, 30)
POSITIVE_RGB = (0, 255, 100)
NEGATIVE_RGB = (255, 70, 70)
ERASE_RGB = (70, 210, 255)
EXCLUDED_RGB = (190, 90, 255)
WORKSPACE_CONTOUR_RGB = (255, 255, 255)
LEGACY_CONTOUR_RGB = (0, 255, 255)
CORE_CONTOUR_RGB = (255, 0, 0)

DIVERGING_CMAP = 'RdBu_r'
DIVERGING_STOPS = ('#2166ac', '#f7f7f7', '#b2182b')
MISSING_COLOR = '#e5e7eb'
SEPARATION_COLOR = '#2166ac'
ZSCORE_LIMITS = (-2.7, 2.7)
SIMILARITY_LIMITS = (0., 1.)
