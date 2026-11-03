/** Browser image-sequence playback and seeking, independent of application startup. */
export function createImageSequenceController({
  $,
  context,
  setPaused,
  pauseSource,
  drain,
  decodeImage,
  fileInfo,
  processFrame,
  captureReference,
  showInputPreview,
  frameFromVideo,
  seekVideo,
  details,
  message,
}) {
  if (typeof context !== 'function') {
    throw new TypeError('Image sequence context is required');
  }
  const state = { images: [], index: 0, task: null, loop: null, seekSerial: 0, delay: null, wake: null };

  function localControls(images) {
    $('solve-range-label').hidden = false;
    $('local-controls').hidden = false;
    $('image-reference-controls').hidden = !images || $('comparison').value === 'consecutive';
    $('image-step-controls').hidden = !images;
    $('image-rate-label').hidden = !images;
    $('video-rate-label').hidden = images;
    $('resolution').disabled = true;
  }

  function updateLocalPosition(input = null) {
    const current = context();
    if (current.source === 'images') {
      const index = input?.index ?? state.index;
      $('local-seek').value = index;
      $('local-position').textContent = `${index + 1} / ${state.images.length} · ${input?.name ?? fileInfo(state.images[index]).name}`;
    } else if (current.source === 'video-file') {
      const time = input?.timeSeconds ?? current.video.currentTime;
      $('local-seek').value = time;
      $('local-position').textContent = `${time.toFixed(3)} / ${Number.isFinite(current.video.duration) ? current.video.duration.toFixed(3) : '?'} s`;
    }
  }

  function wakePlayback() {
    clearTimeout(state.delay);
    state.delay = null;
    state.wake?.();
    state.wake = null;
  }

  function invalidateSeek() {
    state.seekSerial++;
  }

  function reset() {
    state.images = [];
    state.index = 0;
    state.task = null;
    state.loop = null;
    state.seekSerial++;
    wakePlayback();
  }

  async function runImage(index, resetField = false, playback = false) {
    const current = context();
    if (!Number.isInteger(index) || index < 0 || index >= state.images.length) return false;
    const token = current.generation;
    const signal = current.signal;
    await state.task?.catch(() => {});
    if (token !== context().generation) return false;

    const work = (async () => {
      let frame, prior;
      try {
        frame = await decodeImage(state.images[index], [current.referenceInfo.width, current.referenceInfo.height], signal);
        if ($('comparison').value === 'consecutive') {
          if (index === 0) {
            frame.close();
            frame = null;
            state.index = 0;
            updateLocalPosition();
            $('export').disabled = true;
            message('The first image has no predecessor. Choose the next image.');
            return false;
          }
          prior = await decodeImage(state.images[index - 1], [current.referenceInfo.width, current.referenceInfo.height], signal);
        }
        if (token !== context().generation) return false;
        const input = { kind: 'images', index, ...fileInfo(state.images[index]) };
        const referenceInput = prior ? { kind: 'images', index: index - 1, ...fileInfo(state.images[index - 1]) } : null;
        const target = frame;
        const reference = prior;
        frame = prior = null;
        const ok = await processFrame(target, index, {
          force: true,
          reset: resetField,
          input,
          referenceFrame: reference,
          referenceInput,
          playback,
        });
        if (ok) {
          state.index = index;
          updateLocalPosition(input);
        }
        return ok;
      } finally {
        frame?.close();
        prior?.close();
      }
    })();

    state.task = work;
    try {
      return await work;
    } finally {
      if (state.task === work) state.task = null;
    }
  }

  async function setImageReference(index) {
    pauseSource();
    await state.task?.catch(() => {});
    await state.loop?.catch(() => {});
    await drain();
    const current = context();
    const bitmap = await decodeImage(state.images[index], null, current.signal);
    const sourceSettings = context().sourceSettings;
    sourceSettings.reference = { kind: 'images', index, ...fileInfo(state.images[index]) };
    await captureReference(bitmap);
    $('image-reference').value = index;
    $('pause').textContent = 'Start calculation';
    if ($('comparison').value === 'fixed' && state.images.length > 1 && state.index === index) {
      state.index = index + 1 < state.images.length ? index + 1 : index - 1;
    }
    await localSeek(state.index);
    details();
  }

  async function playImages(includeCurrent = false) {
    if (state.loop) {
      await state.loop;
      if (state.loop) return;
    }
    if (includeCurrent) state.index--;
    setPaused(false);
    $('pause').textContent = 'Pause calculation';
    const token = context().generation;
    const operation = (async () => {
      while (!context().paused && token === context().generation) {
        const index = state.index + 1;
        if (index >= state.images.length) break;
        const started = performance.now();
        if (!await runImage(index, index === 0, true)) {
          if (index === 0 && $('comparison').value === 'consecutive') continue;
          break;
        }
        const rate = Number($('image-rate').value);
        if (!Number.isFinite(rate) || rate < 0 || rate > 240) {
          throw new Error('Image rate must be between 0 and 240 (0 = uncapped)');
        }
        if (rate) {
          await new Promise(resolve => {
            state.wake = resolve;
            state.delay = setTimeout(() => {
              state.wake = null;
              state.delay = null;
              resolve();
            }, Math.max(0, 1000 / rate - (performance.now() - started)));
          });
        }
      }
    })();
    state.loop = operation;
    try {
      await operation;
    } catch (error) {
      if (error.name !== 'AbortError') message(error.message, true);
    } finally {
      if (state.loop === operation) {
        state.loop = null;
        if (token === context().generation) {
          setPaused(true);
          $('pause').textContent = 'Start calculation';
        }
      }
    }
  }

  async function localSeek(value) {
    const serial = ++state.seekSerial;
    const token = context().generation;
    pauseSource();
    $('pause').textContent = 'Start calculation';
    await state.task?.catch(() => {});
    await drain();
    if (serial !== state.seekSerial || token !== context().generation) return;

    try {
      const current = context();
      if (current.source === 'images') {
        const index = Number(value);
        const frame = await decodeImage(state.images[index], [current.referenceInfo.width, current.referenceInfo.height], current.signal);
        if (serial !== state.seekSerial || token !== context().generation) {
          frame.close();
          return;
        }
        state.index = index;
        showInputPreview(frame);
        updateLocalPosition();
      } else if (current.source === 'video-file') {
        await seekVideo(current.video, Number(value), current.signal);
        if (serial !== state.seekSerial || token !== context().generation) return;
        const frame = await frameFromVideo();
        if (serial !== state.seekSerial || token !== context().generation) {
          frame.close();
          return;
        }
        showInputPreview(frame);
        updateLocalPosition();
      }
      details();
    } catch (error) {
      if (error.name !== 'AbortError') message(error.message, true);
      updateLocalPosition();
    }
  }

  return {
    get images() { return state.images; },
    get index() { return state.index; },
    set index(value) { state.index = value; },
    get task() { return state.task; },
    get loop() { return state.loop; },
    setImages(images) { state.images = images; state.index = 0; },
    localControls,
    updateLocalPosition,
    wakePlayback,
    invalidateSeek,
    reset,
    runImage,
    setImageReference,
    playImages,
    localSeek,
  };
}
