from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
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

MIN_FRAME_DIMENSION: Final = 16
PARAMETER_ORDER: Final = ("tx", "du_dx", "du_dy", "ty", "dv_dx", "dv_dy")
ImageInput: TypeAlias = str | os.PathLike[str] | bytes | Image.Image | npt.ArrayLike
_MAX_CDP_MESSAGE_BYTES: Final = 128 * 1024 * 1024
_POLL_INTERVAL_SECONDS: Final = 0.05
_CHROMIUM_LOG_TAIL_BYTES: Final = 4_000

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
    device: str
    timer_query: bool
    solver: str = "unknown"
    subgroup_width: int = 0


@dataclass(frozen=True, slots=True)
class ReferenceInfo:
    """Dimensions and separated timings for reference-image preparation."""

    preparation_wall_ms: float
    runtime_build_wall_ms: float
    reference_build_wall_ms: float
    encoding_wall_ms: float
    total_wall_ms: float
    config: Mapping[str, float | int]
    device: str
    grid: tuple[int, int]
    width: int
    height: int
    subset_size: int


@dataclass(slots=True)
class RegistrationResult:
    """Dense affine registration result over the subset grid.

    ``parameters[..., :]`` follows the SYCL-compatible order
    ``[tx, du/dx, du/dy, ty, dv/dx, dv/dy]``.
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
    config: Mapping[str, float | int]
    width: int = 512
    height: int = 512
    subset_size: int = 16

    @property
    def displacement(self) -> np.ndarray:
        """Return the ``(tx, ty)`` displacement field as a zero-copy view."""

        return self.parameters[..., (0, 3)]

    @property
    def accepted_mask(self) -> np.ndarray:
        """Return a mask selecting accepted and median-replaced subsets."""

        return self.status > 0.5

    @property
    def replaced_mask(self) -> np.ndarray:
        """Return a mask selecting subsets replaced by the median filter."""

        return self.status > 1.5

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
        parameters = np.asarray(payload["parameters"], dtype=np.float32).reshape(
            grid_y, grid_x, len(PARAMETER_ORDER)
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
            timing={key: float(value) for key, value in payload.get("timing", {}).items()},
            config=dict(payload.get("config", {})),
            width=int(payload["width"]),
            height=int(payload["height"]),
            subset_size=int(payload["subsetSize"]),
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
    """Small loopback-only HTTP server used by the headless browser."""

    def __init__(self, root: Path) -> None:
        handler = partial(_QuietHandler, directory=str(root))
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.port = int(self._server.server_address[1])
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
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


def _find_chromium(explicit: str | os.PathLike[str] | None) -> str:
    """Resolve a Chromium-compatible executable."""

    if explicit is not None:
        path = Path(explicit).expanduser()
        if not path.is_file() or not os.access(path, os.X_OK):
            raise ICGNError(f"Chromium executable is missing or not executable: {path}")
        return str(path)
    for name in ("chromium-browser", "chromium", "google-chrome", "google-chrome-stable"):
        path = shutil.which(name)
        if path:
            return path
    raise ICGNError("Chromium or Chrome is required, but no executable was found.")


def _as_uint8_gray(image: ImageInput) -> np.ndarray:
    """Normalize a supported image input to a contiguous 8-bit grayscale frame."""

    if isinstance(image, (str, os.PathLike)):
        with Image.open(image) as opened:
            array = np.asarray(opened.convert("L"), dtype=np.uint8)
    elif isinstance(image, bytes):
        with Image.open(io.BytesIO(image)) as opened:
            array = np.asarray(opened.convert("L"), dtype=np.uint8)
    elif isinstance(image, Image.Image):
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
            source = np.clip(source, 0, 255).astype(np.uint8)
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
    ) -> None:
        if startup_timeout <= 0:
            raise ValueError("startup_timeout must be greater than zero")
        if call_timeout <= 0:
            raise ValueError("call_timeout must be greater than zero")
        if backend not in {"gpuio", "three-tsl", "webgpu-wgsl"}:
            raise ValueError("backend must be 'gpuio', 'three-tsl', or 'webgpu-wgsl'")
        assets = _find_runtime_assets(repo_root, backend)
        self.repo_root = assets.root
        self._headless_path = assets.headless_path
        self.backend = backend
        self.chromium = _find_chromium(chromium)
        self.startup_timeout = float(startup_timeout)
        self.call_timeout = float(call_timeout)
        self.no_sandbox = bool(no_sandbox)
        backend_arguments: tuple[str, ...] = ()
        if backend in {"three-tsl", "webgpu-wgsl"}:
            # Chromium on Linux needs all four switches to expose the native
            # Vulkan WebGPU adapter in new-headless mode. Without the surface
            # switch it silently falls back to SwiftShader on this platform.
            backend_arguments = (
                "--enable-unsafe-webgpu",
                "--use-angle=vulkan",
                "--enable-features=Vulkan",
                "--disable-vulkan-surface",
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
        self._server = _StaticServer(self.repo_root)
        self._server.start()
        devtools_port = _free_port()
        self._devtools_origin = f"http://127.0.0.1:{devtools_port}"
        self._temporary = tempfile.TemporaryDirectory(prefix="icgn-webgl-")
        log_path = Path(self._temporary.name) / "chromium.log"
        self._log_handle = log_path.open("wb")
        page_url = f"http://127.0.0.1:{self._server.port}/{self._headless_path}?headless=1"
        if self.backend == "gpuio" and self._requested_dimensions is not None:
            width, height = self._requested_dimensions
            page_url += f"&width={width}&height={height}"
        command = [
            self.chromium,
            "--headless=new",
            "--disable-dev-shm-usage",
            "--use-gl=angle",
            "--use-angle=gl",
            "--ignore-gpu-blocklist",
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
                start_new_session=True,
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
        while time.monotonic() < deadline:
            try:
                if self._evaluate("Boolean(window.ICGNHeadless)"):
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

    @staticmethod
    def _options(
        *,
        iterations: int | None,
        tolerance: float | None,
        zncc_min: float | None,
    ) -> dict[str, Any]:
        options: dict[str, Any] = {}
        if iterations is not None:
            options["iterations"] = int(iterations)
        if tolerance is not None:
            options["tolerance"] = float(tolerance)
        if zncc_min is not None:
            options["znccMin"] = float(zncc_min)
        return options

    def info(self) -> EngineInfo:
        """Return frame, grid, GPU, and timer-query information."""

        payload = self._evaluate("window.ICGNHeadless.info()")
        return EngineInfo(
            width=int(payload["width"]),
            height=int(payload["height"]),
            grid=tuple(map(int, payload["grid"])),
            subset_size=int(payload["subsetSize"]),
            device=str(payload["device"]),
            timer_query=bool(payload["timerQuery"]),
            solver=str(payload.get("solver", "unknown")),
            subgroup_width=int(payload.get("subgroupWidth", 0)),
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
        if self._server is not None:
            self._server.close()
            self._server = None
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
    iterations: int | None = None,
    tolerance: float | None = None,
    zncc_min: float | None = None,
) -> RegistrationResult:
    """Run a one-shot registration and release the headless runtime afterward."""

    with ICGNWebGL(repo_root=repo_root, chromium=chromium, backend=backend) as engine:
        return engine.register_pair(
            reference,
            target,
            iterations=iterations,
            tolerance=tolerance,
            zncc_min=zncc_min,
        )
