import importlib.util
from pathlib import Path


_MODULE_PATH = Path(__file__).resolve().parent / "modules" / "point_cloud_geometry.py"
_SPEC = importlib.util.spec_from_file_location(__name__, _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
globals().update(_MODULE.__dict__)
