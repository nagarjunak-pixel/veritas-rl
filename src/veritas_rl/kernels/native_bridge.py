"""SHA-256 cached C++20 compiler & ctypes bridge for `csrc/veritas_kernels.cpp` with NumPy fallback."""

from __future__ import annotations

import ctypes
import hashlib
import platform
import shutil
import subprocess
from pathlib import Path

import numpy as np
import numpy.typing as npt

F32Array = npt.NDArray[np.float32]
I32Array = npt.NDArray[np.int32]


class VeritasKernelBridge:
    """Compiles and binds `csrc/veritas_kernels.cpp` (`-O3 -std=c++20`) with exact NumPy fallback."""

    def __init__(self, csrc_path: Path | None = None) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        self._src = csrc_path or (repo_root / "csrc" / "veritas_kernels.cpp")
        self._lib: ctypes.CDLL | None = None
        self._backend = "numpy-fallback"
        self._try_compile_and_load()

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def is_native_cpp(self) -> bool:
        return self._lib is not None

    def _try_compile_and_load(self) -> None:
        if not self._src.exists():
            return
        compiler = shutil.which("clang++") or shutil.which("g++")
        if compiler is None:
            return

        cache_dir = self._src.parent.parent / ".veritas_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        src_hash = hashlib.sha256(self._src.read_bytes()).hexdigest()[:12]
        ext = "dylib" if platform.system() == "Darwin" else "so"
        out_lib = cache_dir / f"libveritas_kernels_{src_hash}.{ext}"

        if not out_lib.exists():
            cmd = [
                compiler,
                "-O3",
                "-std=c++20",
                "-shared",
                "-fPIC",
                str(self._src),
                "-o",
                str(out_lib),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if proc.returncode != 0:
                return

        lib = ctypes.CDLL(str(out_lib))
        f_ptr = ctypes.POINTER(ctypes.c_float)
        i_ptr = ctypes.POINTER(ctypes.c_int32)

        lib.veritas_group_relative_advantages_f32.argtypes = [
            f_ptr,
            f_ptr,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_float,
        ]
        lib.veritas_group_relative_advantages_f32.restype = None

        lib.veritas_gae_and_k3_kl_f32.argtypes = [
            f_ptr,
            f_ptr,
            f_ptr,
            f_ptr,
            f_ptr,
            ctypes.c_int32,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            f_ptr,
            f_ptr,
            f_ptr,
            i_ptr,
        ]
        lib.veritas_gae_and_k3_kl_f32.restype = None

        lib.veritas_online_softmax_merge_f32.argtypes = [
            f_ptr,
            f_ptr,
            ctypes.c_int32,
            ctypes.c_int32,
            f_ptr,
            f_ptr,
            f_ptr,
        ]
        lib.veritas_online_softmax_merge_f32.restype = None

        self._lib = lib
        self._backend = f"c++20-native ({Path(compiler).name} -O3)"

    def group_relative_advantages(self, rewards_2d: F32Array, eps: float = 1e-6) -> F32Array:
        arr = np.ascontiguousarray(rewards_2d, dtype=np.float32)
        num_groups, group_size = int(arr.shape[0]), int(arr.shape[1])
        out = np.empty_like(arr)
        if self._lib is not None:
            self._lib.veritas_group_relative_advantages_f32(
                arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                num_groups,
                group_size,
                ctypes.c_float(eps),
            )
            return out

        mean = np.mean(arr, axis=1, keepdims=True)
        var = np.mean((arr - mean) ** 2, axis=1, keepdims=True)
        return np.asarray((arr - mean) / np.sqrt(var + eps), dtype=np.float32)

    def gae_and_k3_kl(
        self,
        rewards: F32Array,
        values: F32Array,
        logp_new: F32Array,
        logp_old: F32Array,
        logp_ref: F32Array,
        *,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps_low: float = 0.20,
        clip_eps_high: float = 0.28,
    ) -> tuple[F32Array, F32Array, F32Array, I32Array]:
        r_c = np.ascontiguousarray(rewards, dtype=np.float32)
        v_c = np.ascontiguousarray(values, dtype=np.float32)
        ln_c = np.ascontiguousarray(logp_new, dtype=np.float32)
        lo_c = np.ascontiguousarray(logp_old, dtype=np.float32)
        lr_c = np.ascontiguousarray(logp_ref, dtype=np.float32)
        length = int(r_c.shape[0])

        gae_out = np.empty(length, dtype=np.float32)
        kl_out = np.empty(length, dtype=np.float32)
        surr_out = np.empty(length, dtype=np.float32)
        clip_flags = np.empty(length, dtype=np.int32)

        if self._lib is not None:
            self._lib.veritas_gae_and_k3_kl_f32(
                r_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                v_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ln_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                lo_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                lr_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                length,
                ctypes.c_float(gamma),
                ctypes.c_float(gae_lambda),
                ctypes.c_float(clip_eps_low),
                ctypes.c_float(clip_eps_high),
                gae_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                kl_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                surr_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                clip_flags.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            )
            return gae_out, kl_out, surr_out, clip_flags

        last_gae = 0.0
        for t in range(length - 1, -1, -1):
            next_v = float(v_c[t + 1]) if (t + 1 < length) else 0.0
            delta = float(r_c[t]) + gamma * next_v - float(v_c[t])
            last_gae = delta + gamma * gae_lambda * last_gae
            gae_out[t] = np.float32(last_gae)

        log_ratio_ref = lr_c.astype(np.float64) - ln_c.astype(np.float64)
        k3 = np.maximum(0.0, np.exp(log_ratio_ref) - log_ratio_ref - 1.0)
        kl_out[:] = k3.astype(np.float32)

        ratio = np.exp(ln_c.astype(np.float64) - lo_c.astype(np.float64))
        clipped_r = np.clip(ratio, 1.0 - clip_eps_low, 1.0 + clip_eps_high)
        adv64 = gae_out.astype(np.float64)
        surr_out[:] = np.minimum(ratio * adv64, clipped_r * adv64).astype(np.float32)
        clip_flags[:] = ((ratio < 1.0 - clip_eps_low) | (ratio > 1.0 + clip_eps_high)).astype(
            np.int32
        )
        return gae_out, kl_out, surr_out, clip_flags

    def online_softmax_merge(
        self,
        scores: F32Array,
        v_block: F32Array,
        out_acc: F32Array,
        m_state: float,
        l_state: float,
    ) -> tuple[F32Array, float, float]:
        block_len = int(scores.shape[0])
        head_dim = int(v_block.shape[1])
        if block_len <= 0:
            return out_acc.copy(), m_state, l_state

        sc_c = np.ascontiguousarray(scores, dtype=np.float32)
        vb_c = np.ascontiguousarray(v_block, dtype=np.float32)
        acc_c = np.ascontiguousarray(out_acc.copy(), dtype=np.float32)

        if self._lib is not None:
            m_arr = np.array([m_state], dtype=np.float32)
            l_arr = np.array([l_state], dtype=np.float32)
            self._lib.veritas_online_softmax_merge_f32(
                sc_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                vb_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                block_len,
                head_dim,
                acc_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                m_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                l_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            )
            return acc_c, float(m_arr[0]), float(l_arr[0])

        m_blk = float(np.max(sc_c))
        m_new = max(m_state, m_blk)
        alpha_old = float(np.exp(m_state - m_new)) if l_state > 0.0 else 0.0
        weights = np.exp(sc_c.astype(np.float64) - m_new)
        new_acc = acc_c.astype(np.float64) * alpha_old + weights @ vb_c.astype(np.float64)
        l_new = l_state * alpha_old + float(np.sum(weights))
        return new_acc.astype(np.float32), m_new, l_new


_GLOBAL_BRIDGE: VeritasKernelBridge | None = None


def get_veritas_bridge() -> VeritasKernelBridge:
    global _GLOBAL_BRIDGE
    if _GLOBAL_BRIDGE is None:
        _GLOBAL_BRIDGE = VeritasKernelBridge()
    return _GLOBAL_BRIDGE
