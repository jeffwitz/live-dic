/** Keep navigation and pending solver edits explicit; no numerical defaults live here. */
export function installWorkspace({ shortcuts }) {
  const buttons = [...document.querySelectorAll('[data-pane]')];
  function showPane(name) {
    for (const button of buttons) {
      const active = button.dataset.pane === name;
      button.toggleAttribute('aria-current', active);
      if (active) button.setAttribute('aria-current', 'step');
      document.getElementById(`pane-${button.dataset.pane}`).hidden = !active;
    }
  }
  buttons.forEach((button, i) => {
    button.onclick = () => showPane(button.dataset.pane);
    button.onkeydown = event => {
      if (!['ArrowLeft', 'ArrowRight'].includes(event.key)) return;
      const next = buttons[(i + (event.key === 'ArrowRight' ? 1 : buttons.length-1)) % buttons.length];
      next.focus(); next.click(); event.preventDefault();
    };
  });
  document.getElementById('help').onclick = () => document.getElementById('quick-guide').showModal();
  window.addEventListener('keydown', event => {
    if (event.repeat || event.ctrlKey || event.altKey || event.metaKey
        || event.target.closest('input,select,textarea,[contenteditable=true],dialog')
        || document.querySelector('#roi.drawing') || document.getElementById('quick-guide').open) return;
    if (event.code === 'Space') { shortcuts.play(); event.preventDefault(); }
    if (event.key.toLowerCase() === 'r') shortcuts.reference();
    if (event.key === 'Escape') shortcuts.clear();
  });
  return { showPane };
}

export function solverOptions() {
  const get = id => document.getElementById(id);
  const base = { profile: get('profile').value };
  if (!get('custom-solver').checked) return base;
  const size = Number(get('subset-size').value), stride = Number(get('subset-stride').value);
  const levels = Number(get('pyramid-levels').value), coarse = Number(get('coarse-iterations').value), fine = Number(get('fine-iterations').value);
  if (!Number.isInteger(stride) || stride < 1 || stride > size) throw new Error('Grid step must be an integer between 1 and the window width');
  if (![coarse, fine].every(v => Number.isInteger(v) && v > 0 && v <= 200)) throw new Error('GN limits must be integers between 1 and 200');
  const temporal = get('temporal-start').checked;
  return { ...base, subsetSize: size, subsetStride: stride, motionModel: get('motion-model').value,
    interpolation: get('interpolation').value, outlierPolicy: get('outlier-policy').value,
    pyramidLevels: levels, iterationsPerLevel: levels === 1 ? [fine] : levels === 2 ? [coarse, fine] : [coarse, 4, fine],
    initialization: temporal ? 'temporal' : 'zero', temporalPrediction: temporal ? 'velocity' : 'none',
    temporalFineCorrection: temporal, divergencePatience: temporal ? 3 : null };
}
