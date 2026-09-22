"""Disk-backed fp16 stitch canvas.

A 4K-safe canvas is ~12 GB of anonymous RAM. Streaming DiT KV on a 24 GB
4090 is charged again against process commit under WDDM. Together they
exceed 64 GB, Windows pages the weights, and later tiles stall
(GPU at 100% and ~110 W). The canvas lives in a file and is mapped only
while a tile is blended or the video is saved, so DiT does not compete
with it. Blend and normalize match the in-memory fp16 path.
"""

from __future__ import annotations

import gc
import os
import shutil
import tempfile
from typing import Iterator, Optional, Tuple

import torch


def release_host_cache() -> None:
    """Return PyTorch's CPU caching allocator blocks to the OS."""
    gc.collect()
    mod = getattr(torch, "_C", None)
    empty = getattr(mod, "_host_emptyCache", None) if mod is not None else None
    if not callable(empty):
        empty = getattr(mod, "_free_host_allocator", None) if mod is not None else None
    if callable(empty):
        try:
            empty()
        except Exception:
            pass
    gc.collect()


def frames_to_fp16_cpu(frames: torch.Tensor, chunk: int = 8) -> torch.Tensor:
    """0–1 fp16 NHWC on CPU.

    Same values as tensor2video(frames).to(float16), but only `chunk` frames
    are promoted to fp32 at a time (a full 1280 tile is ~5 GB in fp32).
    """
    video = frames.detach()
    if video.ndim == 5:
        video = video[0]
    if video.ndim != 4:
        raise ValueError(f"expected C,F,H,W, got {tuple(video.shape)}")
    c, f, h, w = video.shape
    out = torch.empty((f, h, w, c), dtype=torch.float16, device="cpu")
    for i in range(0, f, chunk):
        piece = (video[:, i:i + chunk].float() + 1.0) / 2.0
        out[i:i + chunk].copy_(piece.permute(1, 2, 3, 0).to(dtype=torch.float16, device="cpu"))
        del piece
    return out


def blend_tile_into_canvas(canvas, weights, tile_cpu, mask_nchw, y1, y2, x1, x2):
    mask = mask_nchw.permute(0, 2, 3, 1).to(dtype=torch.float16)
    canvas[:, y1:y2, x1:x2, :] += tile_cpu.to(dtype=torch.float16) * mask
    weights[:, y1:y2, x1:x2, :] += mask


def finalize_stitch_canvas(canvas, weights):
    """In-memory normalize. Tests and any caller that already holds the canvas."""
    n = canvas.shape[0]
    out = torch.empty(canvas.shape, dtype=torch.float32)
    step = 8
    for i in range(0, n, step):
        sl = slice(i, min(i + step, n))
        w = weights[sl].to(torch.float32).clamp_(min=1e-4)
        out[sl] = canvas[sl].to(torch.float32) / w
    return out


def _create_sparse_file(path: str, nbytes: int) -> None:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.argtypes = [
        wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
        wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE,
    ]
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.DeviceIoControl.argtypes = [
        wintypes.HANDLE, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD,
        wintypes.LPVOID, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID,
    ]
    kernel32.DeviceIoControl.restype = wintypes.BOOL
    kernel32.SetFilePointerEx.argtypes = [
        wintypes.HANDLE, ctypes.c_longlong, ctypes.POINTER(ctypes.c_longlong), wintypes.DWORD,
    ]
    kernel32.SetFilePointerEx.restype = wintypes.BOOL
    kernel32.SetEndOfFile.argtypes = [wintypes.HANDLE]
    kernel32.SetEndOfFile.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    GENERIC_WRITE = 0x40000000
    CREATE_ALWAYS = 2
    FILE_ATTRIBUTE_NORMAL = 0x80
    FSCTL_SET_SPARSE = 0x000900C4
    INVALID = wintypes.HANDLE(-1).value

    handle = kernel32.CreateFileW(
        path, GENERIC_WRITE, 0, None, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, None
    )
    if handle == INVALID or handle is None:
        raise OSError(ctypes.get_last_error(), "CreateFileW")
    try:
        returned = wintypes.DWORD(0)
        if not kernel32.DeviceIoControl(
            handle, FSCTL_SET_SPARSE, None, 0, None, 0, ctypes.byref(returned), None
        ):
            raise OSError(ctypes.get_last_error(), "FSCTL_SET_SPARSE")
        newpos = ctypes.c_longlong(0)
        if not kernel32.SetFilePointerEx(handle, ctypes.c_longlong(nbytes), ctypes.byref(newpos), 0):
            raise OSError(ctypes.get_last_error(), "SetFilePointerEx")
        if not kernel32.SetEndOfFile(handle):
            raise OSError(ctypes.get_last_error(), "SetEndOfFile")
    finally:
        kernel32.CloseHandle(handle)


def _ensure_backing_file(path: str, nbytes: int) -> None:
    if nbytes < 0:
        raise ValueError("nbytes must be >= 0")
    if os.name == "nt":
        try:
            _create_sparse_file(path, nbytes)
            return
        except OSError:
            pass
    chunk = 8 * 1024 * 1024
    buf = b"\x00" * min(chunk, max(nbytes, 1))
    with open(path, "wb") as f:
        left = nbytes
        while left > 0:
            n = chunk if left >= chunk else left
            f.write(buf if n == len(buf) else buf[:n])
            left -= n


class DiskStitchCanvas:
    """fp16 color + fp16 weights in temp files. Map them only around a blend."""

    def __init__(self, num_frames: int, height: int, width: int, channels: int, directory: str):
        self.num_frames = int(num_frames)
        self.height = int(height)
        self.width = int(width)
        self.channels = int(channels)
        if min(self.num_frames, self.height, self.width, self.channels) <= 0:
            raise ValueError("canvas dimensions must be positive")
        os.makedirs(directory, exist_ok=True)
        self._dir = tempfile.mkdtemp(prefix="fvsr_stitch_", dir=directory)
        self._color_path = os.path.join(self._dir, "canvas.f16")
        self._weight_path = os.path.join(self._dir, "weights.f16")
        color_n = self.num_frames * self.height * self.width * self.channels
        weight_n = self.num_frames * self.height * self.width
        _ensure_backing_file(self._color_path, color_n * 2)
        _ensure_backing_file(self._weight_path, weight_n * 2)
        self.nbytes = color_n * 2 + weight_n * 2
        self._closed = False

    def _map(self, path: str, shape: Tuple[int, ...]) -> torch.Tensor:
        numel = 1
        for s in shape:
            numel *= int(s)
        flat = torch.from_file(path, shared=True, size=numel, dtype=torch.float16)
        return flat.view(shape)

    def _map_pair(self):
        if self._closed:
            raise RuntimeError("stitch canvas is closed")
        canvas = self._map(
            self._color_path,
            (self.num_frames, self.height, self.width, self.channels),
        )
        weights = self._map(
            self._weight_path,
            (self.num_frames, self.height, self.width, 1),
        )
        return canvas, weights

    def _drop_maps(self, canvas, weights) -> None:
        del canvas, weights
        gc.collect()

    def blend(self, tile_cpu, mask_nchw, y1, y2, x1, x2) -> None:
        canvas, weights = self._map_pair()
        try:
            blend_tile_into_canvas(canvas, weights, tile_cpu, mask_nchw, y1, y2, x1, x2)
        finally:
            self._drop_maps(canvas, weights)

    def blended_chunks(
        self,
        step: int = 4,
        crop: Optional[Tuple[int, int, int, int]] = None,
    ) -> Iterator[torch.Tensor]:
        """Yield float32 frames normalized by weights, `step` frames at a time.

        crop is (top, left, height, width) applied before the fp32 promote so
        the padded canvas is not expanded in RAM.
        """
        canvas, weights = self._map_pair()
        try:
            n = self.num_frames
            for i in range(0, n, step):
                sl = slice(i, min(i + step, n))
                if crop is None:
                    color = canvas[sl]
                    w = weights[sl]
                else:
                    top, left, th, tw = crop
                    color = canvas[sl, top:top + th, left:left + tw, :]
                    w = weights[sl, top:top + th, left:left + tw, :]
                w = w.to(torch.float32).clamp_(min=1e-4)
                yield color.to(torch.float32) / w
        finally:
            self._drop_maps(canvas, weights)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        gc.collect()
        shutil.rmtree(self._dir, ignore_errors=True)
