/** Bounded admission and submission order; no unbounded promise queue. */
export class SlotRing {
  constructor(depth = 1) {
    if (!Number.isInteger(depth) || depth < 1) throw new Error('pipelineDepth must be at least 1');
    this.depth = depth;
    this.active = 0;
    this.tail = Promise.resolve();
  }
  run(operation) {
    if (this.active >= this.depth) return Promise.reject(new Error('Frame backpressure: all slots are busy'));
    this.active++;
    const result = this.tail.then(operation);
    this.tail = result.catch(() => {});
    return result.finally(() => { this.active--; });
  }
  get idle() { return this.active === 0; }
}
