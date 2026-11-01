# Display-cadence controls redirected to bottleneck diagnosis

The owner redirected this task to locating the 4K throughput loss before making
further optimization decisions. The planned twenty-run protocol in
`criteria.json` was stopped after ten consecutive-mode and two fixed-reference
controls. The following run was interrupted explicitly and produced no report.
The full planned matrix is **not qualified**.

Completed raw controls are retained as `cadence-*.json.gz` in
[`../browser-bottleneck-20261101/`](../browser-bottleneck-20261101/README.md).
They show that no-preview runs really execute zero live draws/bitmap transfers
while preserving UHD recording. Their differing solver-call times also expose
run-to-run system variation; do not attribute that variation to display cadence.
The separate stage diagnosis now identifies reference reconstruction, input
transfer, solving, field materialization and storage. No defaults or numerical
shaders were changed to obtain these observations.
