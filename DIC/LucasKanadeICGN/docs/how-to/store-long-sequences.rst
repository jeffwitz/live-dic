Store long browser sequences
============================

This guide concerns the standalone browser workbench. Native Python streaming
continues to return NumPy fields to its caller; it does not use this database.

When upgrading an already open version that used the old 128 MiB in-tab library,
export its results project before reloading. That version's RAM-only fields
cannot be recovered after its tab is closed. Import the project into the new
version to place those fields in its browser database.

Record and reopen
-----------------

1. Load the image folder, video or camera and prepare its reference/ROI.
2. In Results, choose **CPU results cache**. **Automatic** allows up to 1 GiB
   on sufficiently large hosts. Choose **1 GiB** explicitly if appropriate.
   Automatic mode uses the browser's coarse ``deviceMemory`` value, at most
   one eighth of that estimate, capped at 1 GiB and floored at 128 MiB. Unknown
   memory uses 128 MiB. No RAM is reserved by choosing a limit.
3. Enable **Keep computed fields for review**, then start playback/acquisition.
   Fields are committed to IndexedDB from the first saved field, not only when
   RAM is full. This prevents a large first-spill pause and makes eviction safe.
4. Older committed fields leave the RAM cache when required. Use Previous/Next,
   the result-number input or slider to reload any saved field. The dropdown
   shows a nearby slice of the sequence rather than thousands of options.
5. Use **Save analysis in browser** after editing recipes or measurement views.
   The raw computed arrays are already saved; this stores the optional edits.
6. After a reload, choose **List saved sessions**, select a session and
   **Open session**. The catalog loads first, then the selected field. Archived
   sessions cannot receive new acquisition fields. Close the results view to
   start a new recording. Closing a view does not delete its disk data.

The cache retains full typed arrays, including Float64 imported parameters.
No JPEG/PNG encoding, precision reduction or GPU history buffer is introduced.
The reference mask is saved once per session and shared by resident fields.
Histories, post-processing and exports reload data as required. Recording adds
full readback and, by default, an awaited disk transaction, so its sustained throughput needs
separate measurement from GPU-only display. Busy camera/video frames can still
be dropped according to the existing admission policy; this is not a lossless
video recorder. The original images/video must be kept separately.

The opt-in **Experimental pipeline overlap** setting in Solve reserves at most
two recording slots, including a calculation in progress. It can compute the
next field while the previous write completes; it never creates an unbounded
backlog. Pause, source/reference changes and project export drain this queue.
See :doc:`run-in-browser` for activation and preview controls and
:ref:`browser-recording-overlap` for committed throughput and result age.

Make an independent backup
--------------------------

**Save results project** exports a small library in one ``.dicproject`` file.
For a long sequence, use **Download next part** until complete, then download
the parts index. Where the browser supports choosing a writable folder,
**Write all parts to a chosen folder** writes all parts into a new subfolder.
Ordinary downloads remain available without this optional API.

Each ZIP part contains NPY arrays, a JSON manifest, provenance, recipes and a
shared ROI mask. Target size is the smaller of 64 MiB and one quarter of the
cache budget, with at most 128 fields. A single larger field can exceed the
target; the writer refuses a part estimated above 200 MiB. The importer retains
a 256 MiB uncompressed safety cap and 512-field format limit per part. A long
sequence therefore never needs to be materialized in RAM all at once.

Select multiple numbered parts when opening a project. Validation is atomic
per part; previously imported parts remain if a later part fails. A disk error
during admission stops the import: earlier committed fields and the last RAM-only
field remain available. Keep the source part; retrying its import can add duplicate
fields, so inspect the saved counts first. The index
lists SHA-256 hashes, byte sizes and field counts. Keep every part and the
original images. Check actual downloaded files: requesting a download is not
proof that the browser saved it. NumPy reads the arrays with
``allow_pickle=False``. No archive executes code.

Quotas and failure recovery
---------------------------

IndexedDB belongs to the site's origin and browser profile. Changing host,
port, profile or protocol changes which sessions are visible. The application
requests persistent storage when recording is enabled; browsers may deny it.
Clearing site data, deleting a profile, private browsing policies or browser
storage eviction can remove sessions. This is why browser storage is a working
store, not an independent backup. The displayed origin quota is a browser
estimate, not free system disk space and not a promise that allocation succeeds.
The service worker's application-cache update does not delete result sessions.

If a transaction fails, acquisition and recording pause and the completed field
stays in RAM. Do not close the tab. Export the results or resolve the quota and
click **Retry saving RAM-only fields**. Retry commits the retained fields without
duplicating them. No unsaved field is removed to make room. If IndexedDB is
unavailable, recording can use RAM until its budget is reached; the last completed
field is preserved, so this emergency state may exceed the target by one field.
An individual oversized or pinned field may also exceed a very small target.
With experimental overlap enabled, up to two already admitted calculations can
complete; both fields remain in RAM on failure. No subsequent recorded solve is
admitted until recovery. Explicitly re-enable recording after retrying the retained writes.

Concurrent tabs cannot append into the same reopened session. Conflicting
settings saves are rejected explicitly. Another tab deleting a session causes
later operations to fail visibly instead of inventing zero-valued fields.
Deletion is deliberate and permanent; export first.

CPU RAM and GPU memory
----------------------

The CPU cache target covers resident field arrays, their encoded metadata and
shared masks. It is not a limit on the whole browser process. The catalog and
JavaScript objects, one active analysis/worker, decoded frames, renderer,
structured-clone temporaries and bounded export buffers use additional memory.
The catalog still grows with the number of fields; the quota and actual machine
remain finite. This is not a promise of unlimited recording.

**CPU / GPU allocation diagnostics** reports tracked live buffer sizes and known
texture payloads separately from CPU result storage. Readback/mapped GPU API
buffers may reside in host memory. Driver padding, pipelines, compositor and
canvas swapchains are outside that accounting. WebGPU exposes limits, not free
VRAM or reliable physical placement. On an integrated/unified GPU both CPU and
GPU allocations consume shared physical RAM; on a discrete GPU some allocations
and transfers cross pools. The application makes no assumption based on vendor
name and does not derive an imaginary free-VRAM figure from ``maxBufferSize``.

The same IndexedDB/typed-array path is used across systems and needs no native
extension, root access or new dependency. Actual browser/OS storage policies and
GPU drivers still need their own qualification; a Linux run does not certify
Windows, macOS, mobile devices or discrete GPUs.
