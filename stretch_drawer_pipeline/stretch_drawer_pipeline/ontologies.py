import re
import sys
from enum import Enum
from pathlib import Path

_SEMANTIC_ROOT = Path(__file__).resolve().parent.parent.parent / "semantic-object-container-room"
if str(_SEMANTIC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SEMANTIC_ROOT))

from realrobot.voxel_graph_builder import CONTAINER_TYPES


def _camel_to_upper_snake(name):
    return re.sub(r'(?<=[a-z0-9])([A-Z])', r'_\1', name).upper()


DrawerClass = Enum(
    "DrawerClass",
    {_camel_to_upper_snake(t): t for t in sorted(CONTAINER_TYPES)},
    type=str,
)


class HandleClass(str, Enum):
    HANDLE = "Handle"
    KNOB = "Knob"
    DOORKNOB = "Doorknob"
    PULL = "Pull"
    BELLPULL = "Bellpull"
    PULL_CHAIN = "PullChain"


DRAWER_CLASSES = set(DrawerClass)
HANDLE_CLASSES = set(HandleClass)
