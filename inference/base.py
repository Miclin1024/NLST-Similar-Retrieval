import attrs
import data.reader
from definitions import *


@attrs.define()
class _AdapterBase:
    name: str = attrs.field()
    conv: torch.nn.Module = attrs.field()
    reader = data.reader.env_reader
