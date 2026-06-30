import sys
import contextlib
from typing import Any, Iterator, Dict
import lpips

__all__ = ['_silence_console_output', '_safe_print', '_LPIPS_MODEL_CACHE', '_REALESRGAN_UPSAMPLER_CACHE', '_slugify_name']

class _NullStream:
    def write(self, data: Any) -> None:
        pass
    def flush(self) -> None:
        pass

@contextlib.contextmanager
def _silence_console_output() -> Iterator[None]:
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = _NullStream()  # type: ignore
    sys.stderr = _NullStream()  # type: ignore
    try:
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

def _safe_print(*args: Any, **kwargs: Any) -> None:
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    try:
        print(*args, **kwargs)
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

_LPIPS_MODEL_CACHE: Dict[str, lpips.LPIPS] = {}
_REALESRGAN_UPSAMPLER_CACHE: Dict[str, Any] = {}

def _slugify_name(name: str) -> str:
    import re
    slug = name.lower()
    slug = re.sub(r'[^a-z0-9]+', '_', slug)
    return slug.strip('_')

