/** Disk-backed computed fields with a bounded CPU cache. No GPU resources live here. */
import { validateField } from './postprocess.js';
export const MiB = 1024 * 1024;
export const MAX_CACHE_BYTES = 1024 * MiB;
const DATABASE = 'live-dic-results-v1';
const STORES = ['sessions', 'settings', 'fields', 'catalog', 'masks'];
const textBytes = value => new TextEncoder().encode(JSON.stringify(value)).byteLength;
const arraysBytes = field => field.parameters.byteLength + field.status.byteLength + field.iterations.byteLength;

export function defaultCacheBytes(memoryGiB = globalThis.navigator?.deviceMemory) {
  // deviceMemory is coarse, not free RAM. Unknown devices retain a conservative budget.
  return Number.isFinite(memoryGiB) && memoryGiB > 0
    ? Math.min(MAX_CACHE_BYTES, Math.max(128 * MiB, memoryGiB * 1024 * MiB / 8)) : 128 * MiB;
}

function transactionDone(transaction) {
  return new Promise((resolve, reject) => {
    transaction.oncomplete = resolve;
    transaction.onabort = () => reject(transaction.error ?? new Error('Results transaction aborted'));
    transaction.onerror = () => {}; // onabort is the final atomic failure signal.
  });
}
function requestValue(request) {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

/** Typed arrays are structured-cloned, never converted to JSON or lower precision. */
export class IndexedFieldDatabase {
  static async open(factory = globalThis.indexedDB) {
    if (!factory) throw new Error('IndexedDB is unavailable; results can only remain in RAM');
    const request = factory.open(DATABASE, 1);
    request.onupgradeneeded = () => {
      for (const name of STORES) {
        const store = request.result.createObjectStore(name, {keyPath: name === 'sessions' || name === 'settings' ? 'id' : ['session', 'id']});
        if (name !== 'sessions' && name !== 'settings') store.createIndex('session', 'session');
      }
    };
    const db = await new Promise((resolve, reject) => {
      let blocked = false;
      const timer=setTimeout(()=>{blocked=true;reject(new Error('Browser disk storage did not open; RAM-only mode is active until retry'));},10000);
      request.onblocked = () => { clearTimeout(timer);blocked = true; reject(new Error('Another tab blocks results storage; close that tab and retry')); };
      request.onerror = () => {clearTimeout(timer);reject(request.error);};
      request.onsuccess = () => { clearTimeout(timer);if (blocked) request.result.close(); else resolve(request.result); };
    });
    db.onversionchange = () => db.close();
    return new IndexedFieldDatabase(db);
  }
  constructor(db) { this.db = db; }
  async sessions() { return requestValue(this.db.transaction('sessions').objectStore('sessions').getAll()); }
  async catalog(session, map = value => value) {
    // Read index rows one at a time: getAll() can temporarily duplicate a large catalog.
    const request = this.db.transaction('catalog').objectStore('catalog').index('session').openCursor(session);
    return new Promise((resolve, reject) => {
      const rows = []; request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        try { if (request.result) { rows.push(map(request.result.value)); request.result.continue(); } else resolve(rows); }
        catch (error) { reject(error); }
      };
    });
  }
  async settings(session) { return await requestValue(this.db.transaction('settings').objectStore('settings').get(session)) ?? {value: {}, revision: 0}; }
  async read(session, id) { return requestValue(this.db.transaction('fields').objectStore('fields').get([session, id])); }
  async mask(session, id) { return (await requestValue(this.db.transaction('masks').objectStore('masks').get([session, id])))?.data; }
  async write(session, records, masks) {
    const tx = this.db.transaction(['sessions', 'fields', 'catalog', 'masks'], 'readwrite'), done = transactionDone(tx);
    let conflict = false;
    const request = tx.objectStore('sessions').get(session.id);
    request.onsuccess = () => {
      if ((request.result?.count ?? 0) !== session.count - records.length) { conflict = true; tx.abort(); return; }
      for (const mask of masks) tx.objectStore('masks').put({...mask, session: session.id});
      for (const record of records) {
        tx.objectStore('fields').add({session: session.id, id: record.id, field: record.field, metadata: record.metadata, maskId: record.maskId});
        tx.objectStore('catalog').add({...record.descriptor, session: session.id});
      }
      tx.objectStore('sessions').put(session);
    };
    try { await done; } catch (error) { if (conflict) throw new Error('Stored session changed or was deleted in another tab; recording stopped'); throw error; }
  }
  async saveSettings(session, value, revision) {
    const tx = this.db.transaction(['sessions','settings'], 'readwrite'), done = transactionDone(tx);
    let conflict = false, deleted=false;
    const exists=tx.objectStore('sessions').get(session);
    exists.onsuccess=()=>{
      if(!exists.result){deleted=true;tx.abort();return;}
      const request = tx.objectStore('settings').get(session);
      request.onsuccess = () => {
        if ((request.result?.revision ?? 0) !== revision) { conflict = true; tx.abort(); return; }
        tx.objectStore('settings').put({id: session, value, revision: revision + 1});
      };
    };
    try { await done; } catch (error) { if(deleted)throw new Error('This session was deleted in another tab; export retained fields before closing');if (conflict) throw new Error('Another tab changed this session; reopen it before saving settings'); throw error; }
  }
  async remove(session, descriptor, summary) {
    const tx = this.db.transaction(['fields', 'catalog', 'sessions'], 'readwrite'), done = transactionDone(tx);
    let conflict=false;
    const request=tx.objectStore('sessions').get(session);
    request.onsuccess=()=>{
      if(request.result?.count!==summary.count+1){conflict=true;tx.abort();return;}
      for (const name of ['fields', 'catalog']) tx.objectStore(name).delete([session, descriptor.id]);
      tx.objectStore('sessions').put(summary);
    };
    try{await done;}catch(error){if(conflict)throw new Error('Stored session changed in another tab; no field was deleted');throw error;}
  }
  async deleteSession(id) {
    const tx = this.db.transaction(STORES, 'readwrite'), done = transactionDone(tx);
    tx.objectStore('sessions').delete(id); tx.objectStore('settings').delete(id);
    for (const name of ['fields', 'catalog', 'masks']) {
      const store = tx.objectStore(name), cursor = store.index('session').openKeyCursor(id);
      cursor.onsuccess = () => { if (cursor.result) { store.delete(cursor.result.primaryKey); cursor.result.continue(); } };
    }
    await done;
  }
}

function geometry(field) {
  return {width: field.width, height: field.height, grid: field.grid,
    subsetSize: field.subsetSize, subsetStride: field.subsetStride, config: field.config};
}
function compactMetadata(metadata) {
  return {analysisEpoch: metadata.analysisEpoch, archiveSha256: metadata.archiveSha256,
    comparison: metadata.comparison ? {mode: metadata.comparison.mode, target: {
      index: metadata.comparison.target?.index, timeSeconds: metadata.comparison.target?.timeSeconds,
      ordinal: metadata.comparison.target?.ordinal}} : undefined,
    frameIndex: metadata.frameIndex, workbench: metadata.workbench};
}

/** Keeps entry identities stable while evicting only fields already committed to disk. */
export class FieldStorage {
  constructor({memoryLimit = defaultCacheBytes(), openDatabase = () => IndexedFieldDatabase.open()} = {}) {
    this.memoryLimit = memoryLimit; this.openDatabase = openDatabase;
    this.backend = null; this.opening = null; this.warning = null; this.cache = new Map();
    this.maskKeys = new WeakMap(); this.knownMasks = new Set(); this.accountedMasks = new Set(); this.geometries = new Map();
    this.pinned = null; this.session = null; this.readOnly = false; this.queue = Promise.resolve(); this.settingsRevision = 0;
  }
  async database() {
    if (!this.opening) this.opening = this.openDatabase().then(db => this.backend = db).catch(error => { this.warning = error.message; return null; });
    return this.opening;
  }
  async persistence() {
    try { return await globalThis.navigator?.storage?.persist?.() ?? false; } catch { return false; }
  }
  async maskId(mask, field) {
    if (!mask) return null;
    if (!this.maskKeys.has(mask)) {
      // Include geometry even if an imported field has no canonical ROI hash.
      const digest = crypto.subtle.digest('SHA-256', mask).then(bytes =>
        `${field.width}x${field.height}:` + Array.from(new Uint8Array(bytes), value => value.toString(16).padStart(2, '0')).join(''));
      this.maskKeys.set(mask, digest);
    }
    return this.maskKeys.get(mask);
  }
  descriptor(entry) {
    const plain = geometry(entry.field), key = JSON.stringify(plain);
    if (!this.geometries.has(key)) this.geometries.set(key, plain);
    return {id: entry.id, label: entry.label, field: this.geometries.get(key),
      metadata: compactMetadata(entry.metadata), recipe: entry.recipe, bytes: arraysBytes(entry.field)};
  }
  serial(operation) {
    const result = this.queue.then(operation); this.queue = result.catch(() => {}); return result;
  }
  async addMany(entries) {
    return this.serial(async () => {
      if (this.readOnly) throw new Error('This saved session is read-only; start a new results session before recording');
      const db = await this.database(), now = new Date().toISOString();
      const session = this.session ?? {id: crypto.randomUUID(), createdAt: now, count: 0, bytes: 0, nextId: 1, name: 'Computed fields'};
      const masks = [], records = [], residentMasks=new Map();
      for(const entry of this.cache.values())if(entry.mask)residentMasks.set(entry.storage.maskId,entry.mask);
      for (const entry of entries) {
        validateField(entry.field);
        // A narrow view must not retain/serialize an unrelated, much larger backing buffer.
        for(const key of ['parameters','status','iterations']){
          const values=entry.field[key];
          if(!ArrayBuffer.isView(values))throw new Error('Stored computed fields require typed numeric arrays');
          if(values.byteOffset||values.byteLength!==values.buffer.byteLength)entry.field={...entry.field,[key]:values.slice()};
        }
        const maskId = await this.maskId(entry.mask, entry.field), descriptor = {...this.descriptor(entry), maskId};
        if(maskId){if(residentMasks.has(maskId))entry.mask=residentMasks.get(maskId);else residentMasks.set(maskId,entry.mask);}
        records.push({id: entry.id, field: entry.field, metadata: entry.metadata, maskId, descriptor});
        if (maskId && !this.knownMasks.has(maskId) && !masks.some(mask => mask.id === maskId)) masks.push({id: maskId, data: entry.mask});
      }
      const next = {...session, updatedAt: now, count: session.count + entries.length,
        bytes: session.bytes + records.reduce((sum, record) => sum + record.descriptor.bytes, 0) + masks.reduce((sum, mask) => sum + (this.accountedMasks.has(mask.id)?0:mask.data.byteLength), 0),
        nextId: Math.max(session.nextId, ...entries.map(entry => entry.id + 1))};
      let failure = null;
      if (db) { try { await db.write(next, records, masks); } catch (error) { failure = error; } }
      this.session = next;
      masks.forEach(mask=>this.accountedMasks.add(mask.id));
      if (db && !failure) masks.forEach(mask => this.knownMasks.add(mask.id));
      for (let i = 0; i < entries.length; i++) {
        const entry = entries[i]; entry.storage = {saved: !!db && !failure, maskId: records[i].maskId, descriptor: records[i].descriptor};
        this.cache.set(entry.id, entry);
      }
      this.evict();
      if (failure) this.warning = `${failure.name}: ${failure.message}`;
      const unsafe = failure || (!db && this.residentBytes() > this.memoryLimit);
      return {failure: unsafe ? this.warning ?? 'RAM limit reached without disk storage' : null, saved: !!db && !failure};
    });
  }
  residentBytes() {
    const masks = new Set(); let bytes = 0;
    for (const entry of this.cache.values()) {
      bytes += arraysBytes(entry.field) + (entry.metadataBytes ?? textBytes(entry.metadata));
      if (entry.mask && !masks.has(entry.mask)) { masks.add(entry.mask); bytes += entry.mask.byteLength; }
    }
    return bytes;
  }
  evict(protectedId = null) {
    for (const [id, entry] of this.cache) {
      if (this.residentBytes() <= this.memoryLimit) break;
      if (id === this.pinned || id === protectedId || !entry.storage?.saved) continue;
      entry.field = entry.storage.descriptor.field;
      entry.metadata = compactMetadata(entry.metadata); entry.metadataBytes = textBytes(entry.metadata); entry.mask = null;
      this.cache.delete(id);
    }
  }
  async load(entry) {
    await this.queue;
    if (!entry.field.parameters) {
      const db = await this.database(); if (!db) throw new Error('Disk storage is unavailable; this field cannot be loaded');
      const record = await db.read(this.session.id, entry.id);
      if (!record) throw new Error(`Stored field ${entry.id} is missing; no zero field was substituted`);
      validateField(record.field);
      if(JSON.stringify(geometry(record.field))!==JSON.stringify(entry.storage.descriptor.field))throw new Error('Stored field geometry/configuration differs from its catalog');
      // Reuse an already resident static mask instead of copying it per field.
      const shared = [...this.cache.values()].find(value => record.maskId && value.storage?.maskId === record.maskId && value.mask);
      const mask = record.maskId ? shared?.mask ?? await db.mask(this.session.id, record.maskId) : null;
      if (record.maskId && (!mask || mask.length!==record.field.width*record.field.height)) throw new Error('Stored field ROI is missing or has wrong dimensions');
      const view = entry.metadata.workbench;
      entry.field = record.field; entry.metadata = {...record.metadata, ...(view ? {workbench: view} : {})};
      entry.metadataBytes = textBytes(entry.metadata); entry.mask = mask;
    }
    this.cache.delete(entry.id); this.cache.set(entry.id, entry); this.evict(entry.id); return entry;
  }
  pin(id) { this.pinned = id; }
  setLimit(bytes) {
    if (!Number.isSafeInteger(bytes) || bytes < 1024 || bytes > MAX_CACHE_BYTES) throw new Error('Cache limit must be between 1 KiB and 1 GiB');
    this.memoryLimit = bytes; this.evict();
  }
  async flush() { await this.queue; }
  async close() {
    await this.queue;
    const backend=await this.opening;
    backend?.db?.close();
  }
  async saveSettings(entries, selectedId) {
    await this.queue; const db = await this.database();
    if (!db || !this.session) return false;
    if ([...this.cache.values()].some(entry => !entry.storage?.saved)) throw new Error('Some fields remain only in RAM; export them before closing');
    await db.saveSettings(this.session.id, {selectedId, overrides: entries.filter(entry => entry.recipe || entry.metadata.workbench).map(entry => ({id: entry.id, recipe: entry.recipe, workbench: entry.metadata.workbench}))}, this.settingsRevision);
    this.settingsRevision++;
    return true;
  }
  async sessions() { const db = await this.database(); return db ? db.sessions() : []; }
  async openSession(id) {
    const db = await this.database(); if (!db) throw new Error(this.warning);
    const session = (await db.sessions()).find(value => value.id === id); if (!session) throw new Error('Saved session no longer exists');
    this.geometries.clear();
    const catalog = await db.catalog(id, value => {
      const key = JSON.stringify(value.field);
      if (!this.geometries.has(key)) this.geometries.set(key, value.field);
      return {...value, field: this.geometries.get(key)};
    }), saved = await db.settings(id), settings = saved.value;
    if (catalog.length !== session.count) throw new Error('Incomplete session catalog; stored fields were not modified');
    this.cache.clear(); this.knownMasks.clear(); this.accountedMasks=new Set(catalog.map(value=>value.maskId).filter(Boolean)); this.session = session; this.readOnly = true;
    this.settingsRevision = saved.revision;
    const overrides = new Map((settings.overrides ?? []).map(value => [value.id, value]));
    const entries = catalog.sort((a, b) => a.id - b.id).map(value => {
      const override = overrides.get(value.id);
      return {id: value.id, label: value.label, field: value.field, metadata: {...value.metadata, ...(override?.workbench ? {workbench: override.workbench} : {})}, recipe: override?.recipe ?? value.recipe,
        metadataBytes: textBytes(value.metadata), mask: null, storage: {saved: true, maskId: value.maskId, descriptor: value}};
    });
    return {entries, selectedId: settings.selectedId};
  }
  async remove(entry) {
    await this.queue;
    const next = {...this.session, count: this.session.count - 1, bytes: this.session.bytes - entry.storage.descriptor.bytes};
    if (entry.storage?.saved) {
      if (this.readOnly) throw new Error('Close this saved session to start a new one; deleting individual archived fields is disabled');
      if([...this.cache.values()].some(value=>!value.storage?.saved))throw new Error('Retry disk saving or export RAM-only fields before deleting committed data');
      await this.backend.remove(this.session.id, entry.storage.descriptor, next);
    } else if(entry.mask&&entry.storage.maskId&&!this.knownMasks.has(entry.storage.maskId)&&![...this.cache.values()].some(value=>value.id!==entry.id&&value.storage?.maskId===entry.storage.maskId)) {
      next.bytes-=entry.mask.byteLength;
      this.accountedMasks.delete(entry.storage.maskId);
    }
    this.session=next;this.cache.delete(entry.id);
  }
  async deleteSession(id) {
    await this.queue; if (id === this.session?.id) throw new Error('Close the current session before deleting its disk data');
    const db = await this.database(); if (!db) throw new Error(this.warning); await db.deleteSession(id);
  }
  async retry() {
    return this.serial(async () => {
      if (!this.backend) { this.opening = null; await this.database(); }
      if (!this.backend) throw new Error(this.warning);
      const pending = [...this.cache.values()].filter(entry => !entry.storage?.saved);
      if(pending.length){
        // A RAM-only sequence can be large. Commit bounded batches, not a 1 GiB clone.
        let committed=(await this.backend.sessions()).find(value=>value.id===this.session.id);
        if((committed?.count??0)!==this.session.count-pending.length)throw new Error('Stored session changed or was deleted; export retained RAM fields before closing');
        const target=Math.max(256*1024,Math.min(64*MiB,this.memoryLimit/4));
        while(pending.length){
          const batch=[],masks=[],records=[];let bytes=0;
          while(pending.length){
            const entry=pending[0],maskId=entry.storage.maskId;
            const needsMask=maskId&&!this.knownMasks.has(maskId)&&!masks.some(mask=>mask.id===maskId);
            const cost=entry.storage.descriptor.bytes+(needsMask?entry.mask.byteLength:0);
            if(batch.length&&(bytes+cost>target||batch.length===128))break;
            pending.shift();batch.push(entry);bytes+=cost;
            if(needsMask)masks.push({id:maskId,data:entry.mask});
            records.push({id:entry.id,field:entry.field,metadata:entry.metadata,maskId,descriptor:entry.storage.descriptor});
          }
          const summary={...this.session,count:(committed?.count??0)+batch.length,bytes:(committed?.bytes??0)+bytes};
          await this.backend.write(summary,records,masks);committed=summary;
          batch.forEach(entry=>entry.storage.saved=true);masks.forEach(mask=>this.knownMasks.add(mask.id));this.evict();
        }
      }
      this.warning = null; this.evict();
    });
  }
  async quota() {
    try {
      const estimate = await globalThis.navigator?.storage?.estimate?.();
      return {usage: estimate?.usage ?? null, quota: estimate?.quota ?? null, persistent: await globalThis.navigator?.storage?.persisted?.() ?? false};
    } catch { return {usage: null, quota: null, persistent: false}; }
  }
  report() {
    return {sessionId: this.session?.id ?? null, diskAvailable: !!this.backend, readOnly: this.readOnly,
      memoryLimitBytes: this.memoryLimit, residentBytes: this.residentBytes(), residentFields: this.cache.size,
      diskFields: this.backend ? (this.session?.count ?? 0) - [...this.cache.values()].filter(entry => !entry.storage?.saved).length : 0, storedPayloadBytes: this.session?.bytes ?? 0,
      unsavedFields: [...this.cache.values()].filter(entry => !entry.storage?.saved).length, warning: this.warning};
  }
}
