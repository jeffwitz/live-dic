/** Two-stage CPU ownership: reserve before computing, retain until admission.
 *
 * The capacity includes the GPU frame currently being computed. A completed
 * record is never evicted here. On a write failure, remaining records go to
 * explicit RAM retention; a failed retention keeps the original object here.
 */
export class RecordingQueue {
  constructor({ write, retain, onFailure = () => {}, capacity = 2, sampleLimit = 256, now = () => performance.now() }) {
    if (typeof write !== 'function' || typeof retain !== 'function' || !Number.isInteger(capacity) || capacity < 1 || capacity > 2) {
      throw new Error('Recording queue requires write/retain functions and capacity 1 or 2');
    }
    this.write = write; this.retain = retain; this.onFailure = onFailure;
    if (!Number.isInteger(sampleLimit) || sampleLimit<1 || sampleLimit>18000) throw new Error('Invalid recording sample limit');
    this.sampleLimit=sampleLimit;
    this.capacity = capacity; this.now = now; this.entries = new Map();
    this.serial = Promise.resolve(); this.waiters = new Set(); this.nextId = 1;
    this.failure = null; this.saved = 0; this.retained = 0; this.peak = 0; this.samples = [];
  }
  get pending() { return this.entries.size; }
  reserve() {
    if (this.failure || this.entries.size >= this.capacity) return null;
    const ticket = this.nextId++;
    this.entries.set(ticket, { ticket, state: 'computing', record: null });
    this.peak = Math.max(this.peak, this.entries.size);
    return ticket;
  }
  async waitReserve() {
    while (!this.failure) {
      const ticket = this.reserve();
      if (ticket !== null) return ticket;
      await this.changed();
    }
    return null;
  }
  changed() { return new Promise(resolve => { this.waiters.add(resolve); }); }
  signal() { for (const resolve of this.waiters) resolve(); this.waiters.clear(); }
  release(ticket) {
    const entry = this.entries.get(ticket);
    // A caller's finally block may release an unsuccessful GPU admission, but
    // must never release a record already owned by this queue.
    if (entry?.state === 'computing') { this.entries.delete(ticket); this.signal(); }
  }
  fail(error) {
    if (!this.failure) {
      this.failure = error instanceof Error ? error : new Error(String(error));
      try { this.onFailure(this.failure); } catch { /* Retention must still run. */ }
    }
    this.signal();
  }
  submit(ticket, record) {
    const entry = this.entries.get(ticket);
    if (!entry || entry.state !== 'computing') throw new Error('Unknown or already submitted recording ticket');
    entry.record = record; entry.state = 'queued'; entry.receivedAt = this.now();
    const operation = this.serial.then(async () => {
      const started = this.now(); entry.state = 'writing';
      let retained = !!this.failure;
      try {
        if (!retained) {
          const result = await this.write(record, { ticket, waitMs: started - entry.receivedAt });
          // A storage layer may admit the field in RAM while reporting that
          // its transaction failed. It already owns this field; do not add it twice.
          if (result?.failure) { this.fail(new Error(result.failure)); retained = true; }
        } else await this.retain(record);
      } catch (error) {
        this.fail(error); retained = true;
        try { await this.retain(record); }
        catch (retentionError) { entry.state = 'held'; entry.error = retentionError; this.signal(); throw retentionError; }
      }
      const finished = this.now();
      if (retained) this.retained++; else this.saved++;
      this.samples.push({ ticket, retained, admittedEpochMs:performance.timeOrigin+finished, waitMs: started - entry.receivedAt, admissionMs: finished - started,
        receiveToAdmissionMs: finished - entry.receivedAt,
        workerToAdmissionMs: Number.isFinite(record.workerArrivalEpochMs)
          ? (performance.timeOrigin + finished) - record.workerArrivalEpochMs : null });
      if (this.samples.length > this.sampleLimit) this.samples.shift();
      this.entries.delete(ticket); this.signal();
    });
    // Keep the serial tail usable after an unexpected retention failure. The
    // held object remains in entries and drain() explicitly reports it.
    this.serial = operation.catch(() => {});
    return operation;
  }
  async drain() {
    for (;;) {
      await this.serial;
      if ([...this.entries.values()].some(entry => entry.state === 'held')) {
        throw new Error('A completed field remains in the pending RAM queue; retention failed. Do not close this page.');
      }
      if (!this.entries.size) return;
      await this.changed();
    }
  }
  resume() {
    if (this.entries.size) throw new Error('Drain pending recorded fields before resuming');
    this.failure = null; this.signal();
  }
  report() {
    const buffers = new Set(); let bytes = 0, records = 0;
    for (const entry of this.entries.values()) if (entry.record) {
      records++;
      for (const value of [...Object.values(entry.record.field), entry.record.mask]) {
        if (ArrayBuffer.isView(value) && !buffers.has(value.buffer)) { buffers.add(value.buffer); bytes += value.buffer.byteLength; }
      }
    }
    return { capacity: this.capacity, pending: this.entries.size, pendingRecords: records, pendingArrayBytes: bytes,
      peakPending: this.peak, admitted: this.saved, retainedAfterFailure: this.retained,
      blocked: !!this.failure, failure: this.failure?.message ?? null, samples: this.samples.slice(),
      scope: 'Bounded CPU staging, including a reserved GPU frame. Static masks may also be shared with the library; not VRAM.' };
  }
}
