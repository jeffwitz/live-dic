/** Keep the Solve pane informed without requesting any additional GPU data. */
export function imageSequenceTargetCount({ imageCount, currentIndex, includeCurrent, range, comparison }) {
  if (range === 'single') return comparison === 'consecutive' && currentIndex === 0 ? 0 : 1;
  const first = currentIndex + (includeCurrent ? 0 : 1);
  const firstEligible = comparison === 'consecutive' ? Math.max(first, 1) : first;
  return Math.max(0, imageCount - firstEligible);
}

export function createCalculationProgress({ getElement }) {
  const container = getElement('solve-progress-ui');
  const bar = getElement('solve-progress-bar');
  const detail = getElement('solve-progress-detail');
  let active = false;
  let completed = 0;
  let total = null;
  let label = '';

  function render() {
    container.hidden = !label;
    bar.hidden = !active && total === null;
    if (total === null) {
      bar.removeAttribute('value');
      detail.textContent = active
        ? `${label}${completed ? ` · ${completed} frame(s) processed` : ''}`
        : `Calculation ended after ${completed} frame(s).`;
      return;
    }

    bar.max = Math.max(1, total);
    bar.value = Math.min(completed, total);
    detail.textContent = total === 0
      ? 'No target frames are available for this selection.'
      : active
      ? `${label} · ${completed} / ${total} frame(s)`
      : completed >= total
        ? `Calculation complete · ${completed} / ${total} frame(s)`
        : `Calculation stopped · ${completed} / ${total} frame(s)`;
  }

  return {
    begin(phase) {
      active = true;
      completed = 0;
      total = null;
      label = phase;
      render();
    },
    setPhase(phase) {
      if (!active) return;
      label = phase;
      render();
    },
    setTargets(frameCount, phase = 'Calculating fields') {
      if (!active) return;
      total = Number.isInteger(frameCount) && frameCount >= 0 ? frameCount : null;
      label = phase;
      render();
    },
    advance() {
      if (!active) return;
      completed++;
      render();
    },
    finish() {
      if (!active) return;
      active = false;
      render();
    },
    reset() {
      active = false;
      completed = 0;
      total = null;
      label = '';
      render();
    },
    isActive() {
      return active;
    },
    snapshot() {
      return { active, completed, total, label };
    },
  };
}
