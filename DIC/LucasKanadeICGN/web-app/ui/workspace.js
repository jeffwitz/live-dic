/** Keep navigation and pending solver edits explicit; no numerical defaults live here. */
export function installWorkspace({ shortcuts, onPane = () => {} }) {
  let pane = 'source';
  const buttons = [...document.querySelectorAll('[data-pane]')];
  function showPane(name) {
    pane = name;
    for (const button of buttons) {
      const active = button.dataset.pane === name;
      button.toggleAttribute('aria-current', active);
      if (active) button.setAttribute('aria-current', 'step');
      document.getElementById(`pane-${button.dataset.pane}`).hidden = !active;
    }
    onPane(name);
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
        || event.target.closest('button,a,summary,input,select,textarea,[contenteditable=true],[role=button],dialog')
        || document.querySelector('#roi.drawing') || document.getElementById('quick-guide').open) return;
    if (event.code === 'Space') { shortcuts.play(); event.preventDefault(); }
    if (event.key.toLowerCase() === 'r') shortcuts.reference();
    if (event.key === 'Escape') shortcuts.clear();
  });
  return { showPane, get pane() { return pane; } };
}

export function solverOptions() {
  const get = id => document.getElementById(id);
  const base = { profile: get('profile').value };
  if (get('gpu-reference-preparation')?.checked) base.referencePreparation = 'gpu';
  if (get('display-rate') && get('preview-width')) base.supervision = {overlapRecording:!!get('overlap-pipeline')?.checked,
    displayRate:get('display-rate').value==='off'?0:Number(get('display-rate').value),
    previewEnabled:get('display-rate').value!=='off',previewMaxWidth:Number(get('preview-width').value)};
  const fourier = !!get('fourier-start')?.checked;
  const globalAffine = !!get('global-affine-start')?.checked;
  if (fourier && globalAffine) throw new Error('Choose either Fourier or global affine initialization');
  if (fourier) Object.assign(base,{initialization:'fourier',temporalPrediction:'none',temporalFineCorrection:false,divergencePatience:null});
  if (globalAffine) Object.assign(base,{initialization:'global-affine',globalAffineLevels:Number(get('global-affine-levels')?.value??5),motionModel:'affine',temporalPrediction:'none',temporalFineCorrection:false,divergencePatience:null});
  if (!get('custom-solver').checked) return base;
  const size = Number(get('subset-size').value), stride = Number(get('subset-stride').value);
  const levels = Number(get('pyramid-levels').value), coarse = Number(get('coarse-iterations').value), fine = Number(get('fine-iterations').value);
  if (!Number.isInteger(stride) || stride < 1 || stride > size) throw new Error('Grid step must be an integer between 1 and the window width');
  if (!Number.isInteger(levels) || levels < 1 || levels > 6) throw new Error('Pyramid levels must be between 1 and 6');
  if (![coarse, fine].every(v => Number.isInteger(v) && v > 0 && v <= 30)) throw new Error('GN limits must be integers between 1 and 30');
  const temporal = !fourier && !globalAffine && get('temporal-start').checked;
  if (globalAffine && get('motion-model').value !== 'affine') throw new Error('Global affine initialization requires the affine motion model');
  return { ...base, subsetSize: size, subsetStride: stride, motionModel: get('motion-model').value,
    interpolation: get('interpolation').value, outlierPolicy: get('outlier-policy').value,
    pyramidLevels: levels, iterationsPerLevel: levels === 1 ? [fine] : [coarse, ...(levels === 3 ? [4] : Array(Math.max(0, levels - 2)).fill(8)), fine],
    initialization: fourier ? 'fourier' : globalAffine ? 'global-affine' : temporal ? 'temporal' : 'zero', temporalPrediction: temporal ? 'velocity' : 'none',
    temporalFineCorrection: temporal && levels === 2, divergencePatience: temporal ? 3 : null };
}

/** Cold preparation excludes presentation controls, which apply immediately. */
export function solverPreparationKey(requested) {
  const {supervision, ...solver} = requested;
  return JSON.stringify({...solver, overlapRecording: supervision?.overlapRecording ?? false});
}
