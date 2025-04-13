"""Feature extractor class for YOLOS."""

import warnings

from transformers.image_transforms import rgb_to_id as _rgb_to_id
from transformers.utils import logging
from transformers.utils.import_utils import requires

from .image_processing_yolos import YolosImageProcessor

logger = logging.get_logger(__name__)


def rgb_to_id(x):
    warnings.warn(
        "rgb_to_id has moved and will not be importable from this module from v5. "
        "Please import from transformers.image_transforms instead.",
        FutureWarning,
    )
    return _rgb_to_id(x)


@requires(backends=("vision",))
class YolosFeatureExtractor(YolosImageProcessor):
    def __init__(self, *args, **kwargs) -> None:
        warnings.warn(
            "The class YolosFeatureExtractor is deprecated and will be removed in version 5 of Transformers. Please"
            " use YolosImageProcessor instead.",
            FutureWarning,
        )
        super().__init__(*args, **kwargs)


__all__ = ["YolosFeatureExtractor"]
