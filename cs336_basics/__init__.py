
import importlib
try:
    from importlib.metadata import version
    __version__ = version("cs336_basics")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"  # fallback if not installed
