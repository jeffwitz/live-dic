"""Opt-in research host for GPU seed selection; not a public engine API.

One slot, fixed reference, translation/full ROI only. All target processing,
candidate scoring, compaction and re-solves stay on one GPU queue. The only
normal readback is the existing final field. Diagnostics are explicit/cold.
"""

import struct
import time
from importlib.resources import files

import numpy as np

from lucas_kanade_icgn import ICGNWGPUFullPyramid
from lucas_kanade_icgn.roi import prepare_roi
from lucas_kanade_icgn.wgpu_backend import _reduce_half


class GpuSpatialPyramid(ICGNWGPUFullPyramid):
    def __init__(
        self,
        *,
        strategy="baseline",
        passes=2,
        sample_step=1,
        failed_only=False,
        min_gain=1e-4,
        **kwargs,
    ):
        if strategy not in (
            "baseline",
            "median",
            "seed-median",
            "spatial",
            "spatial-median",
            "median-candidate",
        ):
            raise ValueError("Unknown research strategy")
        if passes not in range(1, 5) or sample_step not in (1, 2, 4):
            raise ValueError("Use 1..4 passes and sample_step 1, 2 or 4")
        if kwargs.pop("motion_model", "translation") != "translation":
            raise ValueError("GPU seed research is translation-only")
        if kwargs.pop("outlier_policy", "disabled") != "disabled":
            raise ValueError("Keep the final field raw for this research control")
        for option, required in (
            ("initialization", "zero"),
            ("pyramid_policy", "fixed"),
            ("spatial_weighting", "uniform"),
            ("interpolation", "bspline"),
        ):
            if kwargs.get(option, required) != required:
                raise ValueError(f"Research host requires {option}={required!r}")
        if not np.isfinite(min_gain) or min_gain < 0:
            raise ValueError("min_gain must be finite and nonnegative")
        self.research_resources = []
        self.research = []
        self.strategy = strategy
        self.passes = passes
        self.sample_step = sample_step
        self.failed_only = failed_only
        self.min_gain = min_gain
        self.capture_counts = False
        self._counts_ready = False
        super().__init__(motion_model="translation", outlier_policy="disabled", **kwargs)

    def prepare_research(self, reference, *, roi_mask=None, affine=None):
        """Prepare every resource/pipeline before the measured sequence."""
        if self.research:
            raise ValueError("Create a fresh research host to replace its reference")
        if affine is not None and (
            len(affine["linear"]) != 4
            or not np.isfinite([affine["dx"], affine["dy"], *affine["linear"]]).all()
        ):
            raise ValueError("Affine prior must have finite dx, dy and four gradients")
        self.set_reference(reference, roi_mask=roi_mask)
        device, wgpu = self.device, self._wgpu
        module = device.create_shader_module(
            code=files("lucas_kanade_icgn").joinpath("shaders/spatial_seed.wgsl").read_text()
        )
        images = [reference]
        for _ in range(1, self.pyramid_levels):
            images.append(_reduce_half(images[-1]))
        self.has_affine_prior = affine is not None
        for index, (level, slot, image) in enumerate(
            zip(self._levels, self._full_slots[0].level_slots, images, strict=True)
        ):
            level._ensure_seed_resources(slot)
            n = level.grid_width * level.grid_height

            def buffer(size, usage):
                resource = device.create_buffer(size=size, usage=usage)
                self.research_resources.append(resource)
                return resource

            storage = (
                wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC
            )
            snapshot = buffer(n * 32, storage)
            work = buffer((n + 4) * 4, storage | wgpu.BufferUsage.INDIRECT)
            diagnostics = buffer(
                self.passes * 16, wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC
            )
            scores = buffer(n * 16, storage)
            prior = buffer(n * 32, storage)
            initial = np.zeros((level.grid_height, level.grid_width, 8), np.float32)
            if affine is not None:
                y, x = np.indices(initial.shape[:2])
                scale = 1 << index
                x = (
                    scale * (x * self.subset_stride + (self.subset_size - 1) / 2)
                    + (scale - 1) / 2
                    - (reference.shape[1] - 1) / 2
                )
                y = (
                    scale * (y * self.subset_stride + (self.subset_size - 1) / 2)
                    + (scale - 1) / 2
                    - (reference.shape[0] - 1) / 2
                )
                b = affine["linear"]
                initial[..., 0] = (affine["dx"] + b[0] * x + b[1] * y) / scale
                initial[..., 3] = (affine["dy"] + b[2] * x + b[3] * y) / scale
            device.queue.write_buffer(prior, 0, initial)
            device.queue.write_buffer(slot.buffers["seed_state"], 0, initial)
            plan = level.roi or prepare_roi(
                np.ones(image.shape, bool), self.subset_size, self.subset_stride
            )
            links = plan.neighbour_bits.copy()
            links[~(plan.selected & level.reference_valid)] = 0
            # Coarsest seeded solves have no incoming transfer to initialize
            # their activity mask. Fine masks will be overwritten by transfer.
            device.queue.write_buffer(
                slot.buffers["active_mask"], 0, plan.selected.astype(np.uint32)
            )
            link_buffer = device.create_buffer_with_data(data=links, usage=wgpu.BufferUsage.STORAGE)
            self.research_resources.append(link_buffer)
            texture = device.create_texture(
                size=(level.width, level.height, 1),
                format="r8unorm",
                usage=wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.TEXTURE_BINDING,
            )
            self.research_resources.append(texture)
            device.queue.write_texture(
                {"texture": texture},
                np.ascontiguousarray(image),
                {"bytes_per_row": level.width, "rows_per_image": level.height},
                (level.width, level.height, 1),
            )
            packed = struct.pack(
                "<8I4f",
                level.width,
                level.height,
                level.grid_width,
                n,
                self.subset_size,
                self.subset_stride,
                self.sample_step,
                0,
                self.min_gain,
                1e-3,
                0,
                0,
            )
            config = device.create_buffer_with_data(data=packed, usage=wgpu.BufferUsage.UNIFORM)
            self.research_resources.append(config)
            choose = device.create_compute_pipeline(
                layout="auto",
                compute={
                    "module": module,
                    "entry_point": "choose",
                    "constants": {
                        "FAILED_ONLY": self.failed_only,
                        "CANDIDATE_MODE": 1 if self.strategy == "median-candidate" else 0,
                    },
                },
            )
            finalize = device.create_compute_pipeline(
                layout="auto", compute={"module": module, "entry_point": "finalize"}
            )
            merge = device.create_compute_pipeline(
                layout="auto", compute={"module": module, "entry_point": "retain_accepted"}
            )
            seed_median = device.create_compute_pipeline(
                layout="auto", compute={"module": module, "entry_point": "median_seed"}
            )
            seed_median_group = device.create_bind_group(
                layout=seed_median.get_bind_group_layout(0),
                entries=[
                    self._buffer_resource(k, v)
                    for k, v in (
                        (2, config),
                        (3, snapshot),
                        (4, link_buffer),
                        (5, slot.buffers["filtered_state"]),
                    )
                ],
            )
            resources = [
                {"binding": 0, "resource": texture.create_view()},
                {"binding": 1, "resource": slot.source_texture.create_view()},
                *[
                    self._buffer_resource(k, v)
                    for k, v in (
                        (2, config),
                        (3, slot.buffers["state"] if self.failed_only else snapshot),
                        (4, link_buffer),
                        (5, slot.buffers["seed_state"]),
                        (6, work),
                        (7, scores),
                        (8, prior),
                    )
                ],
            ]
            choose_group = device.create_bind_group(
                layout=choose.get_bind_group_layout(0), entries=resources
            )
            finalize_group = device.create_bind_group(
                layout=finalize.get_bind_group_layout(0), entries=[self._buffer_resource(6, work)]
            )
            merge_group = device.create_bind_group(
                layout=merge.get_bind_group_layout(0),
                entries=[
                    self._buffer_resource(2, config),
                    self._buffer_resource(3, snapshot),
                    self._buffer_resource(5, slot.buffers["state"]),
                ],
            )
            layout = device.create_pipeline_layout(
                bind_group_layouts=[level._seeded_bind_group_layout]
            )
            compact = device.create_compute_pipeline(
                layout=layout,
                compute={
                    "module": level._shader,
                    "entry_point": "solve_compact_translation",
                    "constants": {
                        "MOTION_MODEL": 1,
                        "SUBSET_SIZE": self.subset_size,
                        "SUBSET_STRIDE": self.subset_stride,
                        "SOLVER_WORKGROUP_SIZE": self.solver_workgroup_size,
                        "ROI_ENABLED": int(level.roi is not None and level.roi.effective),
                    },
                },
            )
            compact_group = device.create_bind_group(
                layout=level._seeded_bind_group_layout,
                entries=[
                    {"binding": 0, "resource": slot.source_texture.create_view()},
                    *[
                        self._buffer_resource(k, slot.buffers[name])
                        for k, name in (
                            (1, "config"),
                            (2, "temporary"),
                            (4, "reference"),
                            (6, "metadata"),
                            (7, "state"),
                            (8, "fir"),
                            (9, "filtered_state"),
                            (10, "seed_state"),
                        )
                    ],
                    self._buffer_resource(11, work),
                ],
            )
            median_layout = device.create_pipeline_layout(
                bind_group_layouts=[level._bind_group_layout]
            )
            median = device.create_compute_pipeline(
                layout=median_layout,
                compute={
                    "module": level._shader,
                    "entry_point": "median_filter",
                    "constants": {
                        "MOTION_MODEL": 1,
                        "OUTLIER_POLICY": 3,
                        "MEDIAN_FILTER_SIZE": 3,
                        "SUBSET_SIZE": self.subset_size,
                        "SUBSET_STRIDE": self.subset_stride,
                        "ROI_ENABLED": int(level.roi is not None and level.roi.effective),
                    },
                },
            )
            self.research.append(
                dict(
                    choose=choose,
                    choose_group=choose_group,
                    finalize=finalize,
                    finalize_group=finalize_group,
                    compact=compact,
                    compact_group=compact_group,
                    merge=merge,
                    merge_group=merge_group,
                    snapshot=snapshot,
                    work=work,
                    diagnostics=diagnostics,
                    median=median,
                    scores=scores,
                    reference=texture,
                    prior=prior,
                    seed_median=seed_median,
                    seed_median_group=seed_median_group,
                )
            )
        # Median alters initialization only, never the fine output measurement.
        self.median_edges = {}
        for index, edge in self._full_slots[0].seed_edges.items():
            source = self._full_slots[0].level_slots[index + 1]
            destination = self._full_slots[0].level_slots[index]
            roi = index in self._roi_link_buffers
            self.median_edges[index] = device.create_bind_group(
                layout=self._roi_seed_layout if roi else self._seed_layout,
                entries=[
                    self._buffer_resource(0, source.buffers["filtered_state"]),
                    self._buffer_resource(1, destination.buffers["seed_state"]),
                    self._buffer_resource(2, destination.buffers["active_mask"]),
                    self._buffer_resource(3, edge.config),
                    *([self._buffer_resource(4, self._roi_link_buffers[index])] if roi else []),
                ],
            )

    def _encode_slot(self, slot):
        started = time.perf_counter()
        self._counts_ready = False
        encoder = self.device.create_command_encoder(label="GPU spatial seed research")
        for level_slot in slot.level_slots:
            encoder.clear_buffer(level_slot.buffers["state"])
        if self.has_affine_prior:
            coarse_slot = slot.level_slots[-1]
            prior = self.research[-1]["prior"]
            encoder.copy_buffer_to_buffer(
                prior, 0, coarse_slot.buffers["seed_state"], 0, prior.size
            )
        compute = encoder.begin_compute_pass()
        self._encode_reductions(encoder, slot.reduction_edges, compute_pass=compute)
        for index in range(self.pyramid_levels - 1, -1, -1):
            level = self._levels[index]
            level_slot = slot.level_slots[index]
            r = self.research[index]
            if index < self.pyramid_levels - 1:
                edge = slot.seed_edges[index]
                compute.set_pipeline(
                    self._roi_seed_pipeline
                    if index in self._roi_link_buffers
                    else self._seed_pipeline
                )
                compute.set_bind_group(
                    2,
                    self.median_edges[index]
                    if self.strategy in ("median", "spatial-median", "seed-median")
                    else edge.bind_group,
                )
                compute.dispatch_workgroups((edge.destination_count + 255) // 256)
            seeded = index < self.pyramid_levels - 1 or self.has_affine_prior
            level._encode_registration_passes(
                encoder,
                level_slot,
                level._seeded_solver if seeded else level._portable_solver,
                solver_bind_group=level_slot.seeded_bind_group if seeded else level_slot.bind_group,
                compute_pass=compute,
            )
            compute.end()
            if self.strategy in ("spatial", "spatial-median", "median-candidate"):
                for step in range(self.passes):
                    if not self.failed_only:
                        encoder.copy_buffer_to_buffer(
                            level_slot.buffers["state"], 0, r["snapshot"], 0, r["snapshot"].size
                        )
                    encoder.clear_buffer(r["work"])
                    compute = encoder.begin_compute_pass()
                    compute.set_pipeline(r["choose"])
                    compute.set_bind_group(0, r["choose_group"])
                    compute.dispatch_workgroups(level.grid_width, level.grid_height)
                    compute.set_pipeline(r["finalize"])
                    compute.set_bind_group(0, r["finalize_group"])
                    compute.dispatch_workgroups(1)
                    compute.set_pipeline(r["compact"])
                    compute.set_bind_group(0, r["compact_group"])
                    compute.dispatch_workgroups_indirect(r["work"], 4)
                    if not self.failed_only:
                        compute.set_pipeline(r["merge"])
                        compute.set_bind_group(0, r["merge_group"])
                        compute.dispatch_workgroups(
                            (level.grid_width * level.grid_height + 255) // 256
                        )
                    compute.end()
                    if self.capture_counts:
                        encoder.copy_buffer_to_buffer(r["work"], 0, r["diagnostics"], step * 16, 16)
                        self._counts_ready = True
            if index > 0 and self.strategy == "seed-median":
                encoder.copy_buffer_to_buffer(
                    level_slot.buffers["state"], 0, r["snapshot"], 0, r["snapshot"].size
                )
            compute = encoder.begin_compute_pass()
            if index > 0 and self.strategy == "seed-median":
                compute.set_pipeline(r["seed_median"])
                compute.set_bind_group(0, r["seed_median_group"])
                compute.dispatch_workgroups((level.grid_width * level.grid_height + 63) // 64)
            elif index > 0 and self.strategy in ("median", "spatial-median"):
                compute.set_pipeline(r["median"])
                compute.set_bind_group(0, level_slot.bind_group)
                compute.dispatch_workgroups((level.grid_width * level.grid_height + 255) // 256)
        compute.end()
        fine = slot.level_slots[0]
        encoder.copy_buffer_to_buffer(
            fine.buffers["state"], 0, fine.readback, 0, fine.readback.size
        )
        return encoder.finish(), (time.perf_counter() - started) * 1000

    def read_diagnostics(self):
        """Cold explicit readback, never called by register or the timed loop."""
        if self.strategy in ("baseline", "median", "seed-median"):
            return []
        if not self._counts_ready:
            raise ValueError("Run an explicit capture_counts registration before diagnostics")
        return [
            np.frombuffer(self.device.queue.read_buffer(r["diagnostics"]), np.uint32)
            .reshape(-1, 4)[: self.passes]
            .tolist()
            for r in self.research
        ]

    def register(self, target):
        result = super().register(target)
        result.config = {
            **result.config,
            "spatialSeedResearch": {
                "strategy": self.strategy,
                "extraPassesPerLevel": self.passes
                if self.strategy in ("spatial", "spatial-median", "median-candidate")
                else 0,
                "sampleStep": self.sample_step,
                "failedOnly": self.failed_only,
                "minimumScoreGain": self.min_gain,
                "minimumSeedDisplacementPx": 1e-3,
                "finalFieldRaw": True,
                "notPublicApi": True,
            },
        }
        return result

    def close(self):
        for resource in self.research_resources:
            resource.destroy()
        self.research_resources.clear()
        self.research.clear()
        if hasattr(self, "median_edges"):
            self.median_edges.clear()
        super().close()

    def register_stream(self, frames, **kwargs):
        raise ValueError("The research host supports synchronous register only")
