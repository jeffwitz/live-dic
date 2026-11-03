# Spatial seed candidate research: public UHD control

This is a correctness experiment, not a live-performance qualification. The
CPU oracle uses full readbacks, OpenCV candidate scoring and additional native
IC-GN solves. No numerical engine default or browser option is changed.

## Inputs and configuration

Original archived 3840 × 2160 tensile reference and frame 40, translation model,
32-pixel square subsets, stride 16, B-spline, no spatial post-filter, two levels,
iteration caps 8/16, tolerance 0.005 and ZNCC minimum 0.8. The initial prior is
zero. The interior mask uses the archived 32-pixel margin and contains 30,916
positions. These results do not extend the affine 16/16 qualification to every
subset geometry or motion model.

`uhd-qualification.json` includes both commands, input and script hashes, source
revision/dirty state, actual Intel LNL Vulkan adapter, dependency versions, all
18 field assessments and the stated assessment criteria. The working-tree
scripts are identified by SHA-256; the native kernel code was unchanged.

| Case | Interior coverage | Displacement vector RMSE (px) | p95 error (px) |
|---|---:|---:|---:|
| Initial coarse/fine solve | 100% | 0.0101823 | 0.0172286 |
| Two extra coarse and fine continuation solves | 100% | 0.0101881 | 0.0172315 |
| Two extra coarse and fine scored spatial-seed solves | 100% | 0.0101954 | 0.0172289 |

Both extra-pass cases use the same number of native re-solves and iteration
caps. Spatial mode additionally scores candidate hypotheses on CPU. It is not
a comparison at equal elapsed time or equal total floating-point work.

## Reproduction

From the repository root, run `benchmarks/benchmark_spatial_seed_candidates.py`
under `DIC/LucasKanadeICGN/` with the archived reference/frame-40 paths,
`--mode spatial --output /tmp/seed-spatial`, then repeat with
`--mode continue --output /tmp/seed-continue`. Output directories must be new.
The script defaults to two extra passes per level and exports each complete
NPZ field. See `docs/explanation/spatial-initialization.rst` for the complete
commands and method differences from DIS.

Private large-image explorations and their derived fields remain outside this
repository. Their photometric coverage does not establish metrological accuracy.
ROI discontinuities, noise and production GPU performance require further
qualification before any live integration. No result here establishes 4K60.
