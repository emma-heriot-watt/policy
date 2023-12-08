import logging
from collections.abc import Mapping
from types import MappingProxyType

from emma_common.datamodels import SpeakerRole


logger = logging.getLogger(__name__)


SPEAKER_TOKEN_MAP: Mapping[SpeakerRole, str] = MappingProxyType(
    {SpeakerRole.user: "<<commander>>", SpeakerRole.agent: "<<driver>>"}
)
