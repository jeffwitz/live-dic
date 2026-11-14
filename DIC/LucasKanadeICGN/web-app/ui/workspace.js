/** Keep navigation and pending solver edits explicit; no numerical defaults live here. */
const MAX_SUBSET_SIZE = 46_340;

export function installWorkspace({ shortcuts, onPane = () => {}, onModeChange = () => {} }) {
  let pane = 'source';
  let mode = 'analysis';
  const buttons = [...document.querySelectorAll('[data-pane]')];
  const modeButtons = [...document.querySelectorAll('button[data-workspace-mode]')];

  function showMode(name, { persist = true, destination = null } = {}) {
    if (!['live', 'analysis'].includes(name)) {
      throw new Error("Workspace mode must be 'live' or 'analysis'");
    }
    mode = name;
    document.body.dataset.workspaceMode = name;
    for (const button of modeButtons) {
      button.setAttribute('aria-pressed', String(button.dataset.workspaceMode === name));
    }
    for (const element of document.querySelectorAll('[data-mode-copy]')) {
      element.hidden = element.dataset.modeCopy !== name;
    }
    for (const element of document.querySelectorAll('[data-workspace-only]')) {
      element.hidden = element.dataset.workspaceOnly !== name;
    }
    if (persist) {
      hasSavedMode = true;
      try { localStorage.setItem('live-dic-workspace-mode', name); } catch { /* Storage is optional. */ }
    }
    onModeChange(name);
    if (destination) showPane(destination);
  }

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
  for (const button of modeButtons) {
    button.onclick = () => showMode(button.dataset.workspaceMode, {
      destination: button.dataset.workspacePane || null,
    });
  }
  let savedMode = 'analysis';
  let hasSavedMode = false;
  try {
    const storedMode = localStorage.getItem('live-dic-workspace-mode');
    if (['live', 'analysis'].includes(storedMode)) {
      savedMode = storedMode;
      hasSavedMode = true;
    }
  } catch { /* Storage is optional. */ }
  showMode(savedMode, { persist: false });

  buttons.forEach((button, i) => {
    button.onclick = () => showPane(button.dataset.pane);
    button.onkeydown = event => {
      if (!['ArrowLeft', 'ArrowRight'].includes(event.key)) return;
      const next = buttons[(i + (event.key === 'ArrowRight' ? 1 : buttons.length-1)) % buttons.length];
      next.focus(); next.click(); event.preventDefault();
    };
  });
  document.getElementById('help').onclick = () => document.getElementById('quick-guide').showModal();
  const outlierPolicy = document.getElementById('outlier-policy');
  const medianSize = document.getElementById('median-filter-size');
  const medianScope = document.getElementById('median-filter-scope');
  const updateMedianSizeAvailability = () => {
    medianSize.disabled = outlierPolicy.value !== 'median';
    if (medianScope) medianScope.disabled = outlierPolicy.value !== 'median';
  };
  outlierPolicy.addEventListener('change', updateMedianSizeAvailability);
  updateMedianSizeAvailability();
  window.addEventListener('keydown', event => {
    if (event.repeat || event.ctrlKey || event.altKey || event.metaKey
        || event.target.closest('button,a,summary,input,select,textarea,[contenteditable=true],[role=button],dialog')
        || document.querySelector('#roi.drawing') || document.getElementById('quick-guide').open) return;
    if (event.code === 'Space') { shortcuts.play(); event.preventDefault(); }
    if (event.key.toLowerCase() === 'r') shortcuts.reference();
    if (event.key === 'Escape') shortcuts.clear();
  });
  return {
    showPane,
    showMode,
    get pane() { return pane; },
    get mode() { return mode; },
    get hasSavedMode() { return hasSavedMode; },
  };
}

/** Read the selected initializer and local solver controls without coupling them. */
export function solverOptions(mode = document.body?.dataset.workspaceMode ?? 'analysis') {
  const get = id => document.getElementById(id);
  const base = { profile: 'portable' };
  if (get('display-rate') && get('preview-width')) base.supervision = {overlapRecording:!!get('overlap-pipeline')?.checked,
    displayRate:get('display-rate').value==='off'?0:Number(get('display-rate').value),
    previewEnabled:get('display-rate').value!=='off',previewMaxWidth:Number(get('preview-width').value)};
  if (mode === 'live') {
    return { ...base, initialization: 'temporal', temporalPrediction: 'velocity',
      temporalFineCorrection: true, divergencePatience: 3 };
  }
  if (mode !== 'analysis') throw new Error("Workspace mode must be 'live' or 'analysis'");
  base.roiPolicy = get('roi-policy')?.value ?? 'partial';
  if (get('gpu-reference-preparation')?.checked) base.referencePreparation = 'gpu';
  const fourier = !!get('fourier-start')?.checked;
  const globalAffine = !!get('global-affine-start')?.checked;
  const wgpuMatching = !!get('wgpu-matching-start')?.checked;
  if ([fourier, globalAffine, wgpuMatching].filter(Boolean).length > 1) throw new Error('Choose one image-pair initializer');
  if (fourier) Object.assign(base,{initialization:'fourier',temporalPrediction:'none',temporalFineCorrection:false,divergencePatience:null});
  if (globalAffine) Object.assign(base,{initialization:'global-affine',globalAffineLevels:Number(get('global-affine-levels')?.value??5),temporalPrediction:'none',temporalFineCorrection:false,divergencePatience:null});
  if (wgpuMatching) {
    if (base.referencePreparation === 'gpu') throw new Error('WGPU-Matching initialization requires CPU reference preparation');
    const seedModel = get('wgpu-matching-seed-model')?.value ?? 'global';
    if (!['global', 'local'].includes(seedModel)) throw new Error('WGPU-Matching seed strategy must be global or local');
    Object.assign(base,{initialization:'wgpu-matching',webInitializer:'cpm',temporalPrediction:'none',temporalFineCorrection:false,divergencePatience:null});
    base.wgpuMatchingSeedModel = seedModel;
  }
  const customSolver = !!get('custom-solver')?.checked;
  if (fourier || globalAffine || wgpuMatching || customSolver || base.referencePreparation) base.profile = 'custom';
  const propagationSetting = get('spatial-propagation')?.value ?? 'auto';
  const motionModel = customSolver ? get('motion-model').value : 'affine';
  const interpolation = customSolver ? get('interpolation').value : 'bspline';
  const propagationCompatible = ['translation', 'affine'].includes(motionModel)
    && interpolation === 'bspline' && base.referencePreparation !== 'gpu';
  const spatialPropagationPasses = propagationSetting === 'auto'
    ? (propagationCompatible ? 2 : 0) : Number(propagationSetting);
  if (!Number.isInteger(spatialPropagationPasses) || spatialPropagationPasses < 0 || spatialPropagationPasses > 4) {
    throw new Error('GPU propagation must have zero to four passes per level');
  }
  if (spatialPropagationPasses && (!['translation', 'affine'].includes(motionModel)
      || interpolation !== 'bspline' || base.referencePreparation === 'gpu')) {
    throw new Error('GPU propagation needs Translation or Affine, B-spline and CPU reference preparation; initialization is independent');
  }
  if (spatialPropagationPasses) base.spatialPropagationPasses = spatialPropagationPasses;
  if (propagationSetting !== 'auto') base.profile = 'custom';
  if (!customSolver) return base;
  const size = Number(get('subset-size').value), stride = Number(get('subset-stride').value);
  const levels = Number(get('pyramid-levels').value), coarse = Number(get('coarse-iterations').value), fine = Number(get('fine-iterations').value);
  if (!Number.isInteger(size) || size < 4 || size > MAX_SUBSET_SIZE) {
    throw new Error(`Window width must be an integer from 4 to ${MAX_SUBSET_SIZE} pixels`);
  }
  if (!Number.isInteger(stride) || stride < 1 || stride > size) throw new Error('Grid step must be an integer between 1 and the window width');
  if (!Number.isInteger(levels) || levels < 1 || levels > 6) throw new Error('Pyramid levels must be between 1 and 6');
  if (![coarse, fine].every(v => Number.isInteger(v) && v > 0 && v <= 30)) throw new Error('GN limits must be integers between 1 and 30');
  const temporal = !fourier && !globalAffine && !wgpuMatching && get('temporal-start').checked;
  const outlierPolicy = get('outlier-policy').value;
  const medianFilterSize = outlierPolicy === 'median'
    ? Number(get('median-filter-size')?.value ?? 3) : 3;
  const medianFilterScope = outlierPolicy === 'median'
    ? (get('median-filter-scope')?.value ?? 'rejected') : 'rejected';
  if (![3, 5, 7, 9].includes(medianFilterSize)) throw new Error('Median neighbourhood must be 3, 5, 7, or 9 subsets');
  if (!['rejected', 'all'].includes(medianFilterScope)) throw new Error('Median application must be rejected-only or all-subsets');
  return { ...base, subsetSize: size, subsetStride: stride, motionModel: get('motion-model').value,
    interpolation: get('interpolation').value, outlierPolicy, medianFilterSize, medianFilterScope,
    pyramidLevels: levels, iterationsPerLevel: levels === 1 ? [fine] : [coarse, ...(levels === 3 ? [4] : Array(Math.max(0, levels - 2)).fill(8)), fine],
    initialization: fourier ? 'fourier' : globalAffine ? 'global-affine' : wgpuMatching ? 'wgpu-matching' : temporal ? 'temporal' : 'zero', temporalPrediction: temporal ? 'velocity' : 'none',
    temporalFineCorrection: temporal && levels === 2, divergencePatience: temporal ? 3 : null };
}

/** Cold preparation excludes presentation controls, which apply immediately. */
export function solverPreparationKey(requested) {
  const {supervision, ...solver} = requested;
  return JSON.stringify({...solver, overlapRecording: supervision?.overlapRecording ?? false});
}
