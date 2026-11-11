# 4K seeded-solve transfer diagnostic

Five warm repetitions after two warmups on Intel Graphics LNL, Mesa 25.2.8, Vulkan; reference and target are deterministic synthetic 3840 × 2160 8-bit random speckles with a two-pixel horizontal and one-pixel vertical shift. The 64 × 64 grid has 1,980 centers, 120 initial seeds and 960 candidate frontier windows. The fixed reference is prepared once. All timings are wall times, not GPU timestamp queries; inputs and outputs are materialized NumPy arrays. This case is a simple shift and does not establish accuracy under large or heterogeneous deformation.

| Measured boundary | Median | Range |
| --- | ---: | ---: |
| Frontier call, including buffer allocation, uploads, GPU work and readbacks | 2.589 ms | 2.412–3.298 ms |
| Seeded solve plus final correlation, including target upload and result materialization | 4.892 ms | 4.548–4.957 ms |
| Target texture upload call inside solve; not synchronized separately | 0.372 ms | 0.337–0.465 ms |
| Solve dispatch through state readback | 1.899 ms | 1.765–1.976 ms |
| Additional final correlation, including pipeline creation and readback | 2.346 ms | 2.209–2.348 ms |

The seeded solve accepted 915 of 960 candidates on each repetition. The non-overlapping full-size reference preparation took 964.9 ms in this one run and is excluded from warm calls; it is not a repeated estimate. No global image matcher, anchor selection, multi-pass frontier, quality guard, acquisition or overall initialization time is included.

For a separate diagnostic, a same-sized **buffer**, not a texture, was copied with synchronization. Upload includes a four-byte readback ordering barrier; download is a full synchronous read. These are not measurements of physical bus bandwidth and cannot be directly subtracted from the application times:

| Buffer | Bytes | Upload median | Download median |
| --- | ---: | ---: | ---: |
| Frame-sized | 8,294,400 | 1.229 ms | 0.945 ms |
| Six-parameter state plus status | 63,360 | 0.217 ms | 0.183 ms |

The frontier currently uploads state and permissions and reads seeds and a compact list. The solve uploads the 8-bit target texture, the seed state and active mask, then reads the state. Optional correlation performs another dispatch and score readback. The GPU resources and pipelines for the new frontier and correlation are currently created by each call. This is not GPU-resident across calls; transfers and synchronizations can dominate differently on other hardware or larger grids.

[Raw samples, source hashes and hardware metadata](report.json). Reproduce from the solver directory with `PYTHONPATH=python/src python benchmarks/benchmark_seeded_transfer.py --output /tmp/p4-transfer.json`. The measured shader and source hashes are recorded because this experiment was run on uncommitted changes based on revision `a8c3ca0`.
