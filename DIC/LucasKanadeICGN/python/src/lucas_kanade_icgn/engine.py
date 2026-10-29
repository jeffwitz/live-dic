from __future__ import annotations

import asyncio
import base64
import io
import json
import operator
import os
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from collections.abc import Coroutine, Iterable, Mapping, Sequence
from concurrent.futures import Future
from dataclasses import dataclass
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import TracebackType
from typing import Any, Final, TypeAlias, TypeVar

import numpy as np
import numpy.typing as npt
import websockets
from PIL import Image

from .interpolation import normalize_interpolation
from .motion import AFFINE_PARAMETER_ORDER, normalize_motion_model, parameter_order
from .subset import DEFAULT_SUBSET_SIZE, normalize_subset_size, normalize_subset_stride

MIN_FRAME_DIMENSION: Final = 16
PARAMETER_ORDER: Final = AFFINE_PARAMETER_ORDER
ImageInput: TypeAlias = str | os.PathLike[str] | bytes | Image.Image | npt.ArrayLike
_MAX_CDP_MESSAGE_BYTES: Final = 128 * 1024 * 1024
_POLL_INTERVAL_SECONDS: Final = 0.05
_CHROMIUM_LOG_TAIL_BYTES: Final = 4_000
_MAX_BINARY_RESULT_BYTES: Final = 256 * 1024 * 1024

_T = TypeVar("_T")


class ICGNError(RuntimeError):
    """Raised when the headless runtime or the IC-GN computation fails."""


@dataclass(frozen=True, slots=True)
class EngineInfo:
    """Properties of the active browser GPU runtime."""

    width: int
    height: int
    grid: tuple[int, int]
    subset_size: int
    subset_stride: int
    device: str
    timer_query: bool
    solver: str = "unknown"
    subgroup_width: int = 0
    subgroup_probe_width: int = 0
    subgroup_solver_width: int = 0
    subgroup_validation: str = "unavailable"
    motion_model: str = "affine"


@dataclass(frozen=True, slots=True)
class ReferenceInfo:
    """Dimensions and separated timings for reference-image preparation."""

    preparation_wall_ms: float
    runtime_build_wall_ms: float
    reference_build_wall_ms: float
    encoding_wall_ms: float
    total_wall_ms: float
    config: Mapping[str, float | int | str]
    device: str
    grid: tuple[int, int]
    width: int
    height: int
    subset_size: int
    subset_stride: int


@dataclass(slots=True)
class RegistrationResult:
    """Dense registration result over the subset grid.

    ``parameters[..., :]`` follows the SYCL-compatible order
    ``[tx, du/dx, du/dy, ty, dv/dx, dv/dy]``. The quadratic model appends
    ``[d2u/dx2, d2u/dxdy, d2u/dy2, d2v/dx2, d2v/dxdy, d2v/dy2]``.
    Status values are 1 for accepted, 2 for median-replaced, 0 for
    non-converged, and negative for rejected subsets.
    """

    parameters: np.ndarray
    status: np.ndarray
    iterations: np.ndarray
    accepted: int
    replaced: int
    mean_translation: tuple[float, float] | None
    mean_iterations: float | None
    timing: Mapping[str, float]
    config: Mapping[str, float | int | str]
    width: int = 512
    height: int = 512
    subset_size: int = 16
    subset_stride: int = 16

    @property
    def displacement(self) -> np.ndarray:
        """Return the ``(tx, ty)`` displacement field as a zero-copy view."""

        return self.parameters[..., 0:4:3]

    @property
    def accepted_mask(self) -> np.ndarray:
        """Return a mask selecting accepted and median-replaced subsets."""

        return self.status > 0.5

    @property
    def replaced_mask(self) -> np.ndarray:
        """Return a mask selecting subsets replaced by the median filter."""

        return self.status > 1.5

    @property
    def parameter_order(self) -> tuple[str, ...]:
        """Return the names matching the last axis of :attr:`parameters`."""

        return parameter_order(str(self.config.get("motionModel", "affine")))

    def rmse(self, truth: Sequence[float]) -> float:
        """Compute vector RMSE against a constant ``(dx, dy)`` translation."""

        expected = np.asarray(truth, dtype=np.float64)
        if expected.shape != (2,):
            raise ValueError("truth must contain exactly two values: (dx, dy)")
        mask = self.accepted_mask
        if not np.any(mask):
            return float("nan")
        error = self.displacement[mask].astype(np.float64) - expected
        return float(np.sqrt(np.mean(np.sum(error * error, axis=1))))

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> RegistrationResult:
        """Build a typed result from a JavaScript headless API payload."""

        grid_x, grid_y = map(int, payload["grid"])
        order = parameter_order(str(payload.get("config", {}).get("motionModel", "affine")))
        parameters = np.asarray(payload["parameters"], dtype=np.float32).reshape(
            grid_y, grid_x, len(order)
        )
        status = np.asarray(payload["status"], dtype=np.float32).reshape(grid_y, grid_x)
        iterations = np.asarray(payload["iterations"], dtype=np.float32).reshape(grid_y, grid_x)
        mean = payload.get("meanTranslation")
        return cls(
            parameters=parameters,
            status=status,
            iterations=iterations,
            accepted=int(payload["accepted"]),
            replaced=int(payload["replaced"]),
            mean_translation=(
                None if not mean or mean[0] is None else (float(mean[0]), float(mean[1]))
            ),
            mean_iterations=(
                None if payload.get("meanIterations") is None else float(payload["meanIterations"])
            ),
            timing={
                key: float(value)
                for key, value in payload.get("timing", {}).items()
                if value is not None
            },
            config=dict(payload.get("config", {})),
            width=int(payload["width"]),
            height=int(payload["height"]),
            subset_size=int(payload["subsetSize"]),
            subset_stride=int(payload.get("subsetStride", payload["subsetSize"])),
        )

    @classmethod
    def from_packed_browser_result(
        cls,
        raw: bytes,
        payload: Mapping[str, Any],
    ) -> RegistrationResult:
        """Build a result from the browser's interleaved binary state field."""

        grid_x, grid_y = map(int, payload["grid"])
        order = parameter_order(str(payload.get("config", {}).get("motionModel", "affine")))
        state_stride = int(payload.get("stateStride", 16 if len(order) == 12 else 8))
        expected_bytes = grid_x * grid_y * state_stride * np.dtype(np.float32).itemsize
        if len(raw) != expected_bytes:
            raise ICGNError(
                "Browser binary result size mismatch: "
                f"expected {expected_bytes} bytes, received {len(raw)}"
            )
        # bytearray keeps the public NumPy field writable while all component
        # arrays remain zero-copy views of one compact allocation.
        packed = np.frombuffer(bytearray(raw), dtype="<f4").reshape(grid_y, grid_x, state_stride)
        parameter_count = len(order)
        parameters = packed[..., :parameter_count]
        status = packed[..., parameter_count]
        iterations = packed[..., parameter_count + 1]
        accepted_mask = status > 0.5
        accepted = int(np.count_nonzero(accepted_mask))
        mean_translation = None
        mean_iterations = None
        if accepted:
            mean_translation = (
                float(np.mean(parameters[..., 0][accepted_mask], dtype=np.float64)),
                float(np.mean(parameters[..., 3][accepted_mask], dtype=np.float64)),
            )
            mean_iterations = float(np.mean(iterations[accepted_mask], dtype=np.float64))
        return cls(
            parameters=parameters,
            status=status,
            iterations=iterations,
            accepted=accepted,
            replaced=int(np.count_nonzero(status > 1.5)),
            mean_translation=mean_translation,
            mean_iterations=mean_iterations,
            timing={
                key: float(value)
                for key, value in payload.get("timing", {}).items()
                if value is not None
            },
            config=dict(payload.get("config", {})),
            width=int(payload["width"]),
            height=int(payload["height"]),
            subset_size=int(payload["subsetSize"]),
            subset_stride=int(payload.get("subsetStride", payload["subsetSize"])),
        )


@dataclass(slots=True)
class BenchmarkResult:
    """Timing distribution and final registration from a benchmark run."""

    repeats: int
    warmups: int
    gpu_times_ms: np.ndarray
    wall_times_ms: np.ndarray
    gpu_median_ms: float | None
    gpu_p95_ms: float | None
    wall_median_ms: float
    wall_p95_ms: float
    device: str
    registration: RegistrationResult

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> BenchmarkResult:
        """Build a typed benchmark result from a JavaScript API payload."""

        return cls(
            repeats=int(payload["repeats"]),
            warmups=int(payload["warmups"]),
            gpu_times_ms=np.asarray(payload["gpuTimes"], dtype=np.float64),
            wall_times_ms=np.asarray(payload["wallTimes"], dtype=np.float64),
            gpu_median_ms=(
                None if payload["gpuMedianMs"] is None else float(payload["gpuMedianMs"])
            ),
            gpu_p95_ms=None if payload["gpuP95Ms"] is None else float(payload["gpuP95Ms"]),
            wall_median_ms=float(payload["wallMedianMs"]),
            wall_p95_ms=float(payload["wallP95Ms"]),
            device=str(payload["device"]),
            registration=RegistrationResult.from_payload(payload["registration"]),
        )


class _QuietHandler(SimpleHTTPRequestHandler):
    """Serve solver assets without writing one log line per HTTP request."""

    def log_message(self, _format: str, *args: object) -> None:
        return


class _StaticServer:
    """Loopback asset server with ephemeral binary frame/result exchange."""

    def __init__(self, root: Path) -> None:
        self.transport_token = secrets.token_urlsafe(24)
        self._inputs: dict[str, tuple[memoryview, int, int, str]] = {}
        self._input_queue: deque[str] = deque()
        self._results: dict[str, bytes] = {}
        self._errors: dict[str, str] = {}
        self._condition = threading.Condition()
        self._closed = False
        transport = self

        class _TransportHandler(_QuietHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self) -> None:
                if transport._serve_next_binary_input(self):
                    return
                if transport._serve_binary_input(self):
                    return
                super().do_GET()

            def do_POST(self) -> None:
                if transport._receive_binary_result(self):
                    return
                if transport._receive_binary_failure(self):
                    return
                self.send_error(404)

        handler = partial(_TransportHandler, directory=str(root))
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.port = int(self._server.server_address[1])
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def _transport_ticket(self, path: str, operation: str) -> str | None:
        prefix = f"/__icgn_binary__/{self.transport_token}/{operation}/"
        if not path.startswith(prefix):
            return None
        ticket = path[len(prefix) :]
        if not ticket or "/" in ticket or "?" in ticket:
            return None
        return ticket

    def _serve_binary_input(self, handler: SimpleHTTPRequestHandler) -> bool:
        ticket = self._transport_ticket(handler.path, "input")
        if ticket is None:
            return False
        with self._condition:
            entry = self._inputs.pop(ticket, None)
        if entry is None:
            handler.send_error(404, "Unknown or expired binary input")
            return True
        self._send_binary_input(handler, ticket, entry)
        return True

    def _serve_next_binary_input(self, handler: SimpleHTTPRequestHandler) -> bool:
        path = f"/__icgn_binary__/{self.transport_token}/next"
        if handler.path != path:
            return False
        with self._condition:
            entry: tuple[memoryview, int, int, str] | None = None
            ticket: str | None = None
            while entry is None and not self._closed:
                while self._input_queue and entry is None:
                    ticket = self._input_queue.popleft()
                    entry = self._inputs.pop(ticket, None)
                if entry is None:
                    self._condition.wait()
            if entry is None or ticket is None:
                handler.send_response(204)
                handler.send_header("Content-Length", "0")
                handler.send_header("Cache-Control", "no-store")
                handler.end_headers()
                return True
        self._send_binary_input(handler, ticket, entry)
        return True

    @staticmethod
    def _send_binary_input(
        handler: SimpleHTTPRequestHandler,
        ticket: str,
        entry: tuple[memoryview, int, int, str],
    ) -> None:
        payload, width, height, options = entry
        handler.send_response(200)
        handler.send_header("Content-Type", "application/octet-stream")
        handler.send_header("Content-Length", str(payload.nbytes))
        handler.send_header("Cache-Control", "no-store")
        handler.send_header("X-ICGN-Ticket", ticket)
        handler.send_header("X-ICGN-Width", str(width))
        handler.send_header("X-ICGN-Height", str(height))
        handler.send_header("X-ICGN-Options", options)
        handler.end_headers()
        handler.wfile.write(payload)

    def _receive_binary_result(self, handler: SimpleHTTPRequestHandler) -> bool:
        ticket = self._transport_ticket(handler.path, "result")
        if ticket is None:
            return False
        try:
            length = int(handler.headers.get("Content-Length", "-1"))
        except ValueError:
            length = -1
        if not 0 <= length <= _MAX_BINARY_RESULT_BYTES:
            handler.send_error(413, "Invalid binary result size")
            return True
        payload = handler.rfile.read(length)
        if len(payload) != length:
            handler.send_error(400, "Incomplete binary result")
            return True
        with self._condition:
            self._results[ticket] = payload
            self._condition.notify_all()
        handler.send_response(204)
        handler.send_header("Content-Length", "0")
        handler.send_header("Cache-Control", "no-store")
        handler.end_headers()
        return True

    def _receive_binary_failure(self, handler: SimpleHTTPRequestHandler) -> bool:
        ticket = self._transport_ticket(handler.path, "failure")
        if ticket is None:
            return False
        try:
            length = int(handler.headers.get("Content-Length", "-1"))
        except ValueError:
            length = -1
        if not 0 <= length <= 64 * 1024:
            handler.send_error(413, "Invalid binary transport failure size")
            return True
        message = handler.rfile.read(length).decode("utf-8", errors="replace")
        with self._condition:
            self._errors[ticket] = message
            self._condition.notify_all()
        handler.send_response(204)
        handler.send_header("Content-Length", "0")
        handler.send_header("Cache-Control", "no-store")
        handler.end_headers()
        return True

    def publish_input(
        self,
        frame: npt.NDArray[np.uint8],
        options: Mapping[str, Any] | None = None,
    ) -> str:
        """Expose one contiguous frame once and return its unguessable ticket."""

        if frame.dtype != np.uint8 or frame.ndim != 2 or not frame.flags.c_contiguous:
            raise ValueError("Binary browser input must be a contiguous 2D uint8 array")
        ticket = secrets.token_urlsafe(18)
        height, width = frame.shape
        serialized_options = json.dumps(options or {}, separators=(",", ":"))
        with self._condition:
            if self._closed:
                raise ICGNError("The browser binary transport is closed")
            self._inputs[ticket] = (
                memoryview(frame).cast("B"),
                width,
                height,
                serialized_options,
            )
            self._input_queue.append(ticket)
            self._condition.notify_all()
        return ticket

    def take_result(self, ticket: str, timeout: float) -> bytes:
        """Wait for and consume one binary result posted by the browser."""

        deadline = time.monotonic() + timeout
        with self._condition:
            while ticket not in self._results and ticket not in self._errors:
                if self._closed:
                    raise ICGNError("The browser binary transport was closed")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ICGNError("Timed out waiting for the browser binary result")
                self._condition.wait(remaining)
            if ticket in self._errors:
                message = self._errors.pop(ticket)
                raise ICGNError(f"Browser binary registration failed: {message}")
            return self._results.pop(ticket)

    def discard_exchange(self, ticket: str) -> None:
        with self._condition:
            self._inputs.pop(ticket, None)
            self._results.pop(ticket, None)
            self._errors.pop(ticket, None)

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._inputs.clear()
            self._input_queue.clear()
            self._results.clear()
            self._errors.clear()
            self._condition.notify_all()
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2)


def _free_port() -> int:
    """Ask the operating system for an available loopback TCP port."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@dataclass(frozen=True, slots=True)
class _RuntimeAssets:
    """Filesystem location and URL path of the browser runtime."""

    root: Path
    headless_path: str


def _direct_runtime_assets(root: Path, backend: str = "three-tsl") -> _RuntimeAssets | None:
    """Return a runtime layout rooted directly at one browser backend."""

    if backend == "gpuio":
        required = ("headless.html", "gpu-io.js", "LucasKanadeICGN.js")
    elif backend == "three-tsl":
        required = (
            "headless.html",
            "LucasKanadeICGNThree.js",
            "vendor/three.webgpu.js",
            "vendor/three.tsl.js",
            "vendor/three.core.js",
        )
    else:
        required = ("headless.html", "LucasKanadeICGNWebGPU.js")
    if all((root / filename).is_file() for filename in required):
        if backend == "three-tsl":
            packaged_common = root.parent / "common/ReferencePreparation.js"
            checkout_common = (
                root.parent / "python/src/lucas_kanade_icgn/web/common/ReferencePreparation.js"
            )
            if packaged_common.is_file() or checkout_common.is_file():
                return _RuntimeAssets(root=root.parent, headless_path="three/headless.html")
        if backend == "webgpu-wgsl":
            package_root = root.parent.parent
            shared_assets = (
                root.parent / "common/ReferencePreparation.js",
                package_root / "shaders/icgn.wgsl",
            )
            if all(path.is_file() for path in shared_assets):
                return _RuntimeAssets(
                    root=package_root,
                    headless_path="web/webgpu/headless.html",
                )
        return _RuntimeAssets(root=root, headless_path="headless.html")
    return None


def _checkout_runtime_assets(root: Path, backend: str = "three-tsl") -> _RuntimeAssets | None:
    """Return a source-checkout runtime layout for the selected backend."""

    relative = Path("DIC/LucasKanadeICGN")
    if backend == "gpuio":
        required = (
            root / "gpu-io.js",
            root / relative / "headless.html",
            root / relative / "LucasKanadeICGN.js",
        )
        if all(path.is_file() for path in required):
            return _RuntimeAssets(root=root, headless_path=str(relative / "headless.html"))
    elif backend == "three-tsl":
        relative /= "three"
        if _direct_runtime_assets(root / relative, backend) is not None:
            return _RuntimeAssets(root=root, headless_path=str(relative / "headless.html"))
    else:
        package_root = relative / "python/src/lucas_kanade_icgn"
        runtime = package_root / "web/webgpu"
        if _direct_runtime_assets(root / runtime, backend) is not None:
            return _RuntimeAssets(
                root=root / package_root, headless_path="web/webgpu/headless.html"
            )
    return None


def _find_runtime_assets(
    explicit: str | os.PathLike[str] | None,
    backend: str = "three-tsl",
) -> _RuntimeAssets:
    """Locate self-contained package assets or an explicitly selected checkout."""

    if explicit is not None:
        root = Path(explicit).expanduser().resolve()
        assets = _direct_runtime_assets(root, backend) or _checkout_runtime_assets(root, backend)
        if assets is None:
            raise ICGNError(
                f"The requested runtime root does not contain the {backend!r} browser assets."
            )
        return assets

    # Wheels install package data as ordinary files. Prefer that stable layout
    # so execution never depends on the current directory or a source checkout.
    package_root = Path(__file__).resolve().parent
    packaged_base = package_root / "web"
    packaged_root = packaged_base
    if backend in {"three-tsl", "webgpu-wgsl"}:
        packaged_root /= "three" if backend == "three-tsl" else "webgpu"
    packaged = _direct_runtime_assets(packaged_root, backend)
    if packaged is not None:
        if backend == "three-tsl":
            return _RuntimeAssets(root=packaged_base, headless_path="three/headless.html")
        return packaged

    # Keep a source-tree fallback for incomplete editable installations and for
    # developers exercising this module before rebuilding package data.
    for candidate in (Path.cwd().resolve(), *Path(__file__).resolve().parents):
        checkout = _checkout_runtime_assets(candidate, backend)
        if checkout is not None:
            return checkout
    raise ICGNError("The packaged browser runtime assets are missing; reinstall the package.")


def _chromium_candidates(platform: str) -> tuple[str, ...]:
    """Return Chromium-family executable candidates for an operating system."""

    if platform == "darwin":
        return (
            "chromium",
            "google-chrome",
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Chromium.app/Contents/MacOS/Chromium",
            "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
        )
    if platform == "win32" or platform.startswith("cygwin"):
        return ("chrome.exe", "msedge.exe", "chromium.exe", "google-chrome.exe")
    return ("chromium-browser", "chromium", "google-chrome", "google-chrome-stable")


def _chromium_base_arguments(platform: str) -> tuple[str, ...]:
    """Return platform-safe arguments shared by all browser GPU backends."""

    if platform.startswith("linux"):
        return (
            "--disable-dev-shm-usage",
            "--use-gl=angle",
            "--use-angle=gl",
            "--ignore-gpu-blocklist",
        )
    return ("--ignore-gpu-blocklist",)


def _chromium_backend_arguments(
    platform: str,
    backend: str,
    force_three_webgl: bool,
) -> tuple[str, ...]:
    """Return browser switches needed by the selected GPU implementation."""

    if backend not in {"three-tsl", "webgpu-wgsl"} or force_three_webgl:
        return ()
    if platform.startswith("linux"):
        # New-headless Chromium otherwise tends to select SwiftShader instead
        # of the native Vulkan adapter on Linux. Other systems must retain
        # Chromium's native Metal or D3D adapter selection.
        return (
            "--enable-unsafe-webgpu",
            "--use-angle=vulkan",
            "--enable-features=Vulkan",
            "--disable-vulkan-surface",
        )
    return ("--enable-unsafe-webgpu",)


def _subprocess_session_arguments(platform: str) -> dict[str, bool]:
    """Return process-session options supported by the host operating system."""

    if platform == "win32" or platform.startswith("cygwin"):
        return {}
    return {"start_new_session": True}


def _find_chromium(
    explicit: str | os.PathLike[str] | None,
    *,
    platform: str | None = None,
) -> str:
    """Resolve a Chromium-compatible executable for the current platform."""

    platform_name = sys.platform if platform is None else platform
    if explicit is not None:
        path = Path(explicit).expanduser()
        is_windows = platform_name == "win32" or platform_name.startswith("cygwin")
        if not path.is_file() or (not is_windows and not os.access(path, os.X_OK)):
            raise ICGNError(f"Chromium executable is missing or not executable: {path}")
        return str(path)
    for candidate in _chromium_candidates(platform_name):
        path = Path(candidate).expanduser()
        if path.is_absolute():
            if path.is_file() and os.access(path, os.X_OK):
                return str(path)
            continue
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    raise ICGNError("Chromium or Chrome is required, but no executable was found.")


def _as_uint8_gray(image: ImageInput) -> np.ndarray:
    """Normalize a supported image input to a contiguous 8-bit grayscale frame."""

    if isinstance(image, (str, os.PathLike)):
        with Image.open(image) as opened:
            if opened.mode == "F" or opened.mode.startswith("I"):
                return _as_uint8_gray(np.asarray(opened))
            array = np.asarray(opened.convert("L"), dtype=np.uint8)
    elif isinstance(image, bytes):
        with Image.open(io.BytesIO(image)) as opened:
            if opened.mode == "F" or opened.mode.startswith("I"):
                return _as_uint8_gray(np.asarray(opened))
            array = np.asarray(opened.convert("L"), dtype=np.uint8)
    elif isinstance(image, Image.Image):
        if image.mode == "F" or image.mode.startswith("I"):
            return _as_uint8_gray(np.asarray(image))
        array = np.asarray(image.convert("L"), dtype=np.uint8)
    else:
        source = np.asarray(image)
        if source.ndim not in (2, 3):
            raise ValueError("Expected a 2D, RGB, or RGBA image.")
        if source.ndim == 3 and source.shape[2] not in (3, 4):
            raise ValueError("Expected an RGB or RGBA image with 3 or 4 channels.")
        if np.issubdtype(source.dtype, np.floating):
            source = np.nan_to_num(source, nan=0.0, posinf=255.0, neginf=0.0)
            scale = 255.0 if source.size and float(np.max(source)) <= 1.0 else 1.0
            source = np.clip(source * scale, 0, 255).astype(np.uint8)
        elif source.dtype != np.uint8:
            if not np.issubdtype(source.dtype, np.integer):
                raise ValueError(f"Unsupported image dtype {source.dtype}.")
            if source.size and (int(np.min(source)) < 0 or int(np.max(source)) > 255):
                raise ValueError(
                    "Integer image values must be in [0, 255]. Convert higher-bit-depth "
                    "images explicitly to uint8 with the same calibrated intensity range "
                    "for the reference and every target."
                )
            source = source.astype(np.uint8)
        if source.ndim == 2:
            # Camera frames normally arrive in this form. Avoid a redundant
            # Pillow image construction and grayscale copy on every frame.
            array = source
        else:
            pil = Image.fromarray(source)
            array = np.asarray(pil.convert("L"), dtype=np.uint8)
    height, width = array.shape
    if height < MIN_FRAME_DIMENSION or width < MIN_FRAME_DIMENSION:
        raise ValueError(
            f"Image dimensions must each be at least {MIN_FRAME_DIMENSION} pixels; "
            f"received {array.shape}."
        )
    return np.ascontiguousarray(array)


def _array_data_url(array: npt.NDArray[np.uint8]) -> str:
    """Encode an already normalized frame as a browser-compatible data URL."""

    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG", optimize=False)
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def _data_url(image: ImageInput) -> str:
    """Normalize and encode a frame as a browser-compatible PNG data URL."""

    return _array_data_url(_as_uint8_gray(image))


def _run_async_blocking(coroutine: Coroutine[Any, Any, _T]) -> _T:
    """Run a coroutine from a script or from an active Jupyter event loop."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)

    result: Future[_T] = Future()

    def runner() -> None:
        try:
            result.set_result(asyncio.run(coroutine))
        except BaseException as error:
            # Future.result() re-raises this exception in the calling thread.
            result.set_exception(error)

    thread = threading.Thread(target=runner, name="icgn-webgl-async", daemon=True)
    thread.start()
    thread.join()
    return result.result()


class ICGNWebGL:
    """Persistent affine IC-GN browser GPU backend with no visible UI.

    The object owns a loopback HTTP server and one headless Chromium process.
    Reuse the same instance for a sequence so the reference preparation and
    browser startup costs are paid only once. Instances are not thread-safe.

    ``no_sandbox`` is enabled by default for container compatibility. Set it to
    ``False`` when Chromium's sandbox is available in the execution environment.

    ``interpolation`` selects ``"bspline"`` (historical default), direct
    ``"linear"``, or direct Keys ``"cubic"`` reconstruction. The deprecated
    GPU-IO backend supports only ``"bspline"``. ``subset_size`` selects an 8,
    16, or 32 pixel square. ``subset_stride`` independently selects the field
    spacing and defaults to ``subset_size``.
    """

    def __init__(
        self,
        *,
        repo_root: str | os.PathLike[str] | None = None,
        chromium: str | os.PathLike[str] | None = None,
        startup_timeout: float = 20.0,
        call_timeout: float = 120.0,
        no_sandbox: bool = True,
        chromium_args: Iterable[str] = (),
        backend: str = "three-tsl",
        force_three_webgl: bool = False,
        pyramid_levels: int | None = None,
        iterations_per_level: Iterable[int] | None = None,
        tolerance_per_level: Iterable[float] | None = None,
        interpolation: str = "bspline",
        motion_model: str = "affine",
        subset_size: int = DEFAULT_SUBSET_SIZE,
        subset_stride: int | None = None,
    ) -> None:
        if startup_timeout <= 0:
            raise ValueError("startup_timeout must be greater than zero")
        if call_timeout <= 0:
            raise ValueError("call_timeout must be greater than zero")
        if backend not in {"gpuio", "three-tsl", "webgpu-wgsl"}:
            raise ValueError("backend must be 'gpuio', 'three-tsl', or 'webgpu-wgsl'")
        if force_three_webgl and backend != "three-tsl":
            raise ValueError("force_three_webgl is supported only by the 'three-tsl' backend")
        self.interpolation = normalize_interpolation(interpolation)
        self.motion_model = normalize_motion_model(motion_model)
        self.subset_size = normalize_subset_size(subset_size)
        self.subset_stride = normalize_subset_stride(
            subset_stride,
            subset_size=self.subset_size,
        )
        if backend == "gpuio" and self.interpolation != "bspline":
            raise ValueError("The deprecated 'gpuio' backend supports only interpolation='bspline'")
        if backend == "gpuio" and self.motion_model != "affine":
            raise ValueError("The deprecated 'gpuio' backend supports only motion_model='affine'")
        if self.motion_model == "quadratic" and backend != "webgpu-wgsl":
            raise ValueError("motion_model='quadratic' currently requires backend='webgpu-wgsl'")
        if self.subset_stride != self.subset_size and backend != "webgpu-wgsl":
            raise ValueError(
                "Overlapping subsets currently require backend='webgpu-wgsl'; "
                "Three.js and GPU-IO require subset_stride=subset_size"
            )
        pyramid_options = (pyramid_levels, iterations_per_level, tolerance_per_level)
        if backend != "webgpu-wgsl" and any(value is not None for value in pyramid_options):
            raise ValueError("Pyramid options are supported only by the 'webgpu-wgsl' backend")
        try:
            level_count = 2 if pyramid_levels is None else operator.index(pyramid_levels)
        except TypeError as error:
            raise ValueError("pyramid_levels must be an integer from 1 to 3") from error
        if isinstance(pyramid_levels, bool) or not 1 <= level_count <= 3:
            raise ValueError("pyramid_levels must be an integer from 1 to 3")
        try:
            raw_iteration_schedule = (
                None if iterations_per_level is None else tuple(iterations_per_level)
            )
            if raw_iteration_schedule is not None and any(
                isinstance(value, bool) for value in raw_iteration_schedule
            ):
                raise TypeError
            iteration_schedule = (
                None
                if raw_iteration_schedule is None
                else tuple(operator.index(value) for value in raw_iteration_schedule)
            )
        except TypeError as error:
            raise ValueError("iterations_per_level values must be integers") from error
        tolerance_schedule = (
            None
            if tolerance_per_level is None
            else tuple(float(value) for value in tolerance_per_level)
        )
        for schedule, name in (
            (iteration_schedule, "iterations_per_level"),
            (tolerance_schedule, "tolerance_per_level"),
        ):
            if schedule is not None and len(schedule) != level_count:
                raise ValueError(f"{name} must contain one value per pyramid level")
        if iteration_schedule is not None and any(
            value < 1 or value > 30 for value in iteration_schedule
        ):
            raise ValueError("iterations_per_level values must be integers between 1 and 30")
        if tolerance_schedule is not None and any(
            not np.isfinite(value) or value <= 0 for value in tolerance_schedule
        ):
            raise ValueError("tolerance_per_level values must be finite and positive")
        assets = _find_runtime_assets(repo_root, backend)
        self.repo_root = assets.root
        self._headless_path = assets.headless_path
        self.backend = backend
        self.force_three_webgl = bool(force_three_webgl)
        self.pyramid_levels = level_count if backend == "webgpu-wgsl" else 1
        self.iterations_per_level = iteration_schedule
        self.tolerance_per_level = tolerance_schedule
        # Browser discovery is intentionally lazy. Pure configuration and
        # numerical tests must not require a browser installation.
        self.chromium = _find_chromium(chromium) if chromium is not None else None
        self.startup_timeout = float(startup_timeout)
        self.call_timeout = float(call_timeout)
        self.no_sandbox = bool(no_sandbox)
        backend_arguments = _chromium_backend_arguments(
            sys.platform,
            backend,
            self.force_three_webgl,
        )
        self.chromium_args = (*backend_arguments, *tuple(chromium_args))
        self._server: _StaticServer | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._temporary: tempfile.TemporaryDirectory[str] | None = None
        self._log_handle: Any = None
        self._websocket_url: str | None = None
        self._devtools_origin: str | None = None
        self._reference_ready = False
        self._requested_dimensions: tuple[int, int] | None = None
        self._runtime_dimensions: tuple[int, int] | None = None

    def start(self) -> ICGNWebGL:
        """Start Chromium and wait until the JavaScript headless API is ready."""

        if self._process is not None:
            if self._process.poll() is None:
                return self
            self.close()
        # Resolve the external executable before allocating any local server,
        # log, or temporary profile so a missing browser cannot leak them.
        chromium = _find_chromium(self.chromium)
        self.chromium = chromium
        self._server = _StaticServer(self.repo_root)
        self._server.start()
        devtools_port = _free_port()
        self._devtools_origin = f"http://127.0.0.1:{devtools_port}"
        self._temporary = tempfile.TemporaryDirectory(prefix="icgn-webgl-")
        log_path = Path(self._temporary.name) / "chromium.log"
        self._log_handle = log_path.open("wb")
        page_url = f"http://127.0.0.1:{self._server.port}/{self._headless_path}?headless=1"
        page_url += f"&subsetSize={self.subset_size}"
        page_url += f"&subsetStride={self.subset_stride}"
        if self.backend != "gpuio":
            page_url += f"&binaryToken={self._server.transport_token}"
        if self.force_three_webgl:
            page_url += "&forceWebGL=1"
        if self.backend == "gpuio" and self._requested_dimensions is not None:
            width, height = self._requested_dimensions
            page_url += f"&width={width}&height={height}"
        command = [
            chromium,
            "--headless=new",
            *_chromium_base_arguments(sys.platform),
            f"--remote-debugging-port={devtools_port}",
            f"--remote-allow-origins={self._devtools_origin}",
            f"--user-data-dir={self._temporary.name}/profile",
            *self.chromium_args,
        ]
        if self.no_sandbox:
            command.append("--no-sandbox")
        command.append(page_url)
        try:
            self._process = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=self._log_handle,
                **_subprocess_session_arguments(sys.platform),
            )
        except Exception:
            self.close()
            raise
        deadline = time.monotonic() + self.startup_timeout
        endpoint = f"{self._devtools_origin}/json/list"
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                break
            try:
                with urllib.request.urlopen(endpoint, timeout=0.5) as response:
                    pages = json.load(response)
                page = next(
                    item
                    for item in pages
                    if item.get("type") == "page"
                    and f"/{self._headless_path}?" in item.get("url", "")
                )
                self._websocket_url = str(page["webSocketDebuggerUrl"])
                if self._wait_until_ready(deadline):
                    if self.backend == "gpuio":
                        self._runtime_dimensions = self._requested_dimensions or (512, 512)
                    return self
            except (
                json.JSONDecodeError,
                urllib.error.URLError,
                TimeoutError,
                StopIteration,
                KeyError,
            ) as error:
                last_error = error
            time.sleep(_POLL_INTERVAL_SECONDS)
        details = ""
        if log_path.exists():
            details = log_path.read_text(errors="replace")[-_CHROMIUM_LOG_TAIL_BYTES:]
        self.close()
        raise ICGNError(
            "The headless browser GPU engine did not start before the timeout. "
            f"Last error: {last_error!r}\n{details}"
        )

    def _wait_until_ready(self, deadline: float) -> bool:
        expression = (
            "Boolean(window.ICGNHeadless)"
            if self.backend == "gpuio"
            else "Boolean(window.ICGNHeadless && window.ICGNBinaryTransport)"
        )
        while time.monotonic() < deadline:
            try:
                if self._evaluate(expression):
                    return True
            except ICGNError:
                pass
            time.sleep(_POLL_INTERVAL_SECONDS)
        return False

    async def _evaluate_async(self, expression: str) -> Any:
        if self._websocket_url is None or self._devtools_origin is None:
            raise ICGNError("The headless engine is not running.")
        # A fresh CDP connection keeps calls independent and avoids sharing an
        # asyncio transport across script and Jupyter event loops.
        async with websockets.connect(
            self._websocket_url,
            origin=self._devtools_origin,
            max_size=_MAX_CDP_MESSAGE_BYTES,
            open_timeout=self.call_timeout,
        ) as websocket:
            request = {
                "id": 1,
                "method": "Runtime.evaluate",
                "params": {
                    "expression": expression,
                    "awaitPromise": True,
                    "returnByValue": True,
                },
            }
            await websocket.send(json.dumps(request))
            while True:
                raw_message = await asyncio.wait_for(websocket.recv(), timeout=self.call_timeout)
                message = json.loads(raw_message)
                if message.get("id") != 1:
                    continue
                if "error" in message:
                    raise ICGNError(f"DevTools error: {message['error']}")
                result = message.get("result", {})
                if "exceptionDetails" in result:
                    exception = result["exceptionDetails"]
                    description = (
                        exception.get("exception", {}).get("description")
                        or exception.get("text")
                        or repr(exception)
                    )
                    raise ICGNError(f"JavaScript error: {description}")
                remote = result.get("result", {})
                if remote.get("subtype") == "error":
                    raise ICGNError(remote.get("description", "Unknown JavaScript error"))
                return remote.get("value")

    def _evaluate(self, expression: str) -> Any:
        if self._process is None:
            self.start()
        try:
            return _run_async_blocking(self._evaluate_async(expression))
        except asyncio.TimeoutError as error:
            raise ICGNError(
                f"The computation exceeded the {self.call_timeout:g} s timeout."
            ) from error
        except (OSError, websockets.exceptions.WebSocketException) as error:
            raise ICGNError(f"DevTools transport failed: {error}") from error

    def _options(
        self,
        *,
        iterations: int | None,
        tolerance: float | None,
        zncc_min: float | None,
    ) -> dict[str, Any]:
        options: dict[str, Any] = {}
        options["interpolation"] = self.interpolation
        options["motionModel"] = self.motion_model
        if iterations is not None:
            options["iterations"] = int(iterations)
        if tolerance is not None:
            options["tolerance"] = float(tolerance)
        if zncc_min is not None:
            options["znccMin"] = float(zncc_min)
        if self.backend == "webgpu-wgsl":
            options["pyramidLevels"] = self.pyramid_levels
            if self.iterations_per_level is not None:
                options["iterationsPerLevel"] = list(self.iterations_per_level)
            if self.tolerance_per_level is not None:
                options["tolerancePerLevel"] = list(self.tolerance_per_level)
        return options

    def info(self) -> EngineInfo:
        """Return frame, grid, GPU, and timer-query information."""

        payload = self._evaluate("window.ICGNHeadless.info()")
        return EngineInfo(
            width=int(payload["width"]),
            height=int(payload["height"]),
            grid=tuple(map(int, payload["grid"])),
            subset_size=int(payload["subsetSize"]),
            subset_stride=int(payload.get("subsetStride", payload["subsetSize"])),
            device=str(payload["device"]),
            timer_query=bool(payload["timerQuery"]),
            solver=str(payload.get("solver", "unknown")),
            subgroup_width=int(payload.get("subgroupWidth", 0)),
            subgroup_probe_width=int(payload.get("subgroupProbeWidth", 0)),
            subgroup_solver_width=int(payload.get("subgroupSolverWidth", 0)),
            subgroup_validation=str(payload.get("subgroupValidation", "unavailable")),
            motion_model=str(payload.get("motionModel", self.motion_model)),
        )

    def set_reference(
        self,
        image: ImageInput,
        *,
        iterations: int | None = None,
        tolerance: float | None = None,
        zncc_min: float | None = None,
    ) -> ReferenceInfo:
        """Upload and precompute all target-independent reference data."""

        total_start = time.perf_counter()
        encoding_start = time.perf_counter()
        array = _as_uint8_gray(image)
        height, width = array.shape
        if self.backend == "gpuio":
            dimensions = (width, height)
            if width % 16 or height % 16:
                raise ValueError(
                    "The GPU-IO backend requires width and height to be multiples of 16; "
                    f"received {width}x{height}."
                )
            if self._runtime_dimensions is not None and self._runtime_dimensions != dimensions:
                # GPU-IO allocates all textures at page creation. Reopening the
                # private headless page is the only dimension change required;
                # the public engine instance and API remain unchanged.
                self.close()
            self._requested_dimensions = dimensions
        data_url = _array_data_url(array)
        encoding_wall_ms = (time.perf_counter() - encoding_start) * 1_000.0
        options = self._options(iterations=iterations, tolerance=tolerance, zncc_min=zncc_min)
        expression = (
            f"window.ICGNHeadless.setReference({json.dumps(data_url)}, {json.dumps(options)})"
        )
        payload = self._evaluate(expression)
        total_wall_ms = (time.perf_counter() - total_start) * 1_000.0
        self._reference_ready = True
        return ReferenceInfo(
            preparation_wall_ms=float(payload["preparationWallMs"]),
            runtime_build_wall_ms=float(payload["runtimeBuildWallMs"]),
            reference_build_wall_ms=float(payload["referenceBuildWallMs"]),
            encoding_wall_ms=encoding_wall_ms,
            total_wall_ms=total_wall_ms,
            config=dict(payload["config"]),
            device=str(payload["device"]),
            grid=tuple(map(int, payload["grid"])),
            width=int(payload["width"]),
            height=int(payload["height"]),
            subset_size=int(payload["subsetSize"]),
            subset_stride=int(payload.get("subsetStride", payload["subsetSize"])),
        )

    def register(
        self,
        target: ImageInput,
        *,
        iterations: int | None = None,
        tolerance: float | None = None,
        zncc_min: float | None = None,
    ) -> RegistrationResult:
        """Register one target against the previously prepared reference."""

        if not self._reference_ready:
            raise ICGNError("Call set_reference() before register().")
        if self.backend != "gpuio":
            return self._register_binary(
                target,
                iterations=iterations,
                tolerance=tolerance,
                zncc_min=zncc_min,
            )
        total_start = time.perf_counter()
        encoding_start = time.perf_counter()
        data_url = _data_url(target)
        encoding_wall_ms = (time.perf_counter() - encoding_start) * 1_000.0
        options = self._options(iterations=iterations, tolerance=tolerance, zncc_min=zncc_min)
        expression = f"window.ICGNHeadless.register({json.dumps(data_url)}, {json.dumps(options)})"
        result = RegistrationResult.from_payload(self._evaluate(expression))
        total_wall_ms = (time.perf_counter() - total_start) * 1_000.0
        result.timing = {
            **result.timing,
            "encodingWallMs": encoding_wall_ms,
            "pythonCallWallMs": total_wall_ms,
        }
        return result

    def _register_binary(
        self,
        target: ImageInput,
        *,
        iterations: int | None,
        tolerance: float | None,
        zncc_min: float | None,
    ) -> RegistrationResult:
        """Register through same-origin raw-byte input and packed output routes."""

        if self._server is None:
            raise ICGNError("The headless engine is not running.")
        total_started = time.perf_counter()
        preparation_started = time.perf_counter()
        frame = _as_uint8_gray(target)
        input_preparation_wall_ms = (time.perf_counter() - preparation_started) * 1_000.0
        options = self._options(iterations=iterations, tolerance=tolerance, zncc_min=zncc_min)
        ticket = self._server.publish_input(frame, options)
        try:
            response = self._server.take_result(ticket, self.call_timeout)
        finally:
            self._server.discard_exchange(ticket)
        if len(response) < 4:
            raise ICGNError("Browser binary response is missing its metadata header")
        metadata_size = int.from_bytes(response[:4], byteorder="little", signed=False)
        metadata_end = 4 + metadata_size
        if metadata_end > len(response):
            raise ICGNError("Browser binary response has truncated metadata")
        try:
            payload = json.loads(response[4:metadata_end].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ICGNError("Browser binary response metadata is invalid") from exc
        raw = response[metadata_end:]
        result_started = time.perf_counter()
        result = RegistrationResult.from_packed_browser_result(raw, payload)
        result_build_wall_ms = (time.perf_counter() - result_started) * 1_000.0
        result.timing = {
            **result.timing,
            "inputPreparationWallMs": input_preparation_wall_ms,
            "resultBuildWallMs": result_build_wall_ms,
            "encodingWallMs": 0.0,
            "pythonCallWallMs": (time.perf_counter() - total_started) * 1_000.0,
        }
        return result

    def register_pair(
        self,
        reference: ImageInput,
        target: ImageInput,
        *,
        iterations: int | None = None,
        tolerance: float | None = None,
        zncc_min: float | None = None,
    ) -> RegistrationResult:
        """Prepare a reference and register one target."""

        self.set_reference(
            reference,
            iterations=iterations,
            tolerance=tolerance,
            zncc_min=zncc_min,
        )
        return self.register(
            target,
            iterations=iterations,
            tolerance=tolerance,
            zncc_min=zncc_min,
        )

    def register_many(
        self,
        reference: ImageInput,
        targets: Iterable[ImageInput],
        *,
        iterations: int | None = None,
        tolerance: float | None = None,
        zncc_min: float | None = None,
    ) -> list[RegistrationResult]:
        """Register multiple targets while preparing the reference only once."""

        self.set_reference(
            reference,
            iterations=iterations,
            tolerance=tolerance,
            zncc_min=zncc_min,
        )
        return [
            self.register(
                target,
                iterations=iterations,
                tolerance=tolerance,
                zncc_min=zncc_min,
            )
            for target in targets
        ]

    def benchmark(
        self,
        reference: ImageInput,
        target: ImageInput,
        *,
        repeats: int = 30,
        warmups: int = 3,
        iterations: int | None = None,
        tolerance: float | None = None,
        zncc_min: float | None = None,
    ) -> BenchmarkResult:
        """Prepare one reference, then benchmark registration of one target."""

        self.set_reference(
            reference,
            iterations=iterations,
            tolerance=tolerance,
            zncc_min=zncc_min,
        )
        return self.benchmark_target(
            target,
            repeats=repeats,
            warmups=warmups,
            iterations=iterations,
            tolerance=tolerance,
            zncc_min=zncc_min,
        )

    def benchmark_target(
        self,
        target: ImageInput,
        *,
        repeats: int = 30,
        warmups: int = 3,
        iterations: int | None = None,
        tolerance: float | None = None,
        zncc_min: float | None = None,
    ) -> BenchmarkResult:
        """Benchmark registration against the already prepared reference.

        Reference construction is deliberately excluded. Use :meth:`set_reference`
        first and retain its :class:`ReferenceInfo` to report preparation time.
        """

        if not self._reference_ready:
            raise ICGNError("Call set_reference() before benchmark_target().")
        if repeats < 1:
            raise ValueError("repeats must be greater than zero")
        if warmups < 0:
            raise ValueError("warmups must not be negative")
        options = self._options(iterations=iterations, tolerance=tolerance, zncc_min=zncc_min)
        data_url = _data_url(target)
        expression = (
            f"window.ICGNHeadless.benchmark({json.dumps(data_url)}, {int(repeats)}, "
            f"{int(warmups)}, {json.dumps(options)})"
        )
        return BenchmarkResult.from_payload(self._evaluate(expression))

    def close(self) -> None:
        """Release the Chromium process, server, log, and temporary profile."""

        # End the long poll before terminating Chromium so the browser transport
        # exits normally instead of reporting a connection failure on shutdown.
        if self._server is not None:
            self._server.close()
            self._server = None
        process, self._process = self._process, None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None
        self._websocket_url = None
        self._devtools_origin = None
        self._reference_ready = False
        self._runtime_dimensions = None

    def __enter__(self) -> ICGNWebGL:
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def register_pair(
    reference: ImageInput,
    target: ImageInput,
    *,
    repo_root: str | os.PathLike[str] | None = None,
    chromium: str | os.PathLike[str] | None = None,
    backend: str = "three-tsl",
    force_three_webgl: bool = False,
    pyramid_levels: int | None = None,
    iterations_per_level: Iterable[int] | None = None,
    tolerance_per_level: Iterable[float] | None = None,
    interpolation: str = "bspline",
    motion_model: str = "affine",
    subset_size: int = DEFAULT_SUBSET_SIZE,
    subset_stride: int | None = None,
    iterations: int | None = None,
    tolerance: float | None = None,
    zncc_min: float | None = None,
) -> RegistrationResult:
    """Run a one-shot registration and release the headless runtime afterward."""

    with ICGNWebGL(
        repo_root=repo_root,
        chromium=chromium,
        backend=backend,
        force_three_webgl=force_three_webgl,
        pyramid_levels=pyramid_levels,
        iterations_per_level=iterations_per_level,
        tolerance_per_level=tolerance_per_level,
        interpolation=interpolation,
        motion_model=motion_model,
        subset_size=subset_size,
        subset_stride=subset_stride,
    ) as engine:
        return engine.register_pair(
            reference,
            target,
            iterations=iterations,
            tolerance=tolerance,
            zncc_min=zncc_min,
        )
