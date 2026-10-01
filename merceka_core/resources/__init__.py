"""Cross-process resource primitives (file locks, GPU coordination)."""

from merceka_core.resources.gpu import gpu_lock, gpu_lock_sync, GPU_LOCK_PATH

__all__ = ["gpu_lock", "gpu_lock_sync", "GPU_LOCK_PATH"]
