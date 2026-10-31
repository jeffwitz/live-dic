/** Render a self-contained presentation export; computed arrays remain untouched. */
export function resultFigure({ field, annotations, title, label, legend, details }) {
  const canvas = document.createElement('canvas');
  canvas.width = Math.max(1600, field.width);
  const scale = canvas.width / 1600, margin = 40 * scale, contentWidth = canvas.width - 2 * margin;
  const imageHeight = Math.round(field.height * canvas.width / field.width);
  const context = canvas.getContext('2d');
  // Prepare text before setting the final height (which resets the canvas state).
  function wrap(text, fontSize) {
    context.font = `${fontSize * scale}px system-ui`;
    // Bound presentation text from imported names; archives retain the complete metadata.
    text = String(text); if (text.length > 1000) text = text.slice(0, 997) + '…';
    const rows = []; let line = '';
    for (const word of text.split(/\s+/)) {
      if (line && context.measureText(line + word).width > contentWidth) { rows.push(line.trimEnd()); line = ''; }
      for (const character of word) {
        if (context.measureText(line + character).width > contentWidth) { rows.push(line.trimEnd()); line = ''; }
        line += character;
      }
      line += ' ';
    }
    if (line.trim()) rows.push(line.trim());
    return rows;
  }
  const sections = [
    { size: 28, rows: wrap(title, 28) },
    { size: 18, rows: wrap(label, 18) },
    ...details.map(text => ({ size: 18, rows: wrap(text, 18) })),
    { size: 16, rows: wrap('Reference pixel indices: x right, y down. Presentation only; retain the NPZ/project for values and provenance. Foundational work by Pierre Margerit.', 16) },
  ];
  const legendHeight = legend.categories ? 120 : 108;
  const textHeight = sections.reduce((height, section) => height + section.rows.length * (section.size + 9) + 12, 0);
  canvas.height = Math.ceil(imageHeight + (legendHeight + textHeight + 35) * scale);
  context.imageSmoothingEnabled = false;
  context.drawImage(field, 0, 0, canvas.width, imageHeight);
  context.drawImage(annotations, 0, 0, canvas.width, imageHeight);
  context.fillStyle = '#fff'; context.fillRect(0, imageHeight, canvas.width, canvas.height - imageHeight);
  let y = imageHeight + 30 * scale;
  context.font = `${18 * scale}px system-ui`;
  if (legend.categories) {
    const columnWidth = contentWidth / 2;
    legend.categories.forEach((category, index) => {
      const x = margin + (index % 2) * columnWidth, rowY = y + Math.floor(index / 2) * 34 * scale;
      context.fillStyle = category.colour; context.fillRect(x, rowY, 24 * scale, 20 * scale);
      context.fillStyle = '#163540'; context.fillText(category.label, x + 34 * scale, rowY + 17 * scale);
    });
  } else {
    const gradient = context.createLinearGradient(margin, 0, margin + contentWidth, 0);
    legend.stops.forEach(([position, colour]) => gradient.addColorStop(position, colour));
    context.fillStyle = gradient; context.fillRect(margin, y, contentWidth, 24 * scale);
    context.fillStyle = '#163540'; context.fillText(legend.low, margin, y + 51 * scale);
    context.textAlign = 'right'; context.fillText(legend.high, margin + contentWidth, y + 51 * scale); context.textAlign = 'left';
    context.fillStyle = '#475660'; context.fillRect(margin, y + 66 * scale, 18 * scale, 18 * scale);
    context.fillStyle = '#163540'; context.fillText('Unavailable / rejected; colour limits do not clip exported values.', margin + 28 * scale, y + 81 * scale);
  }
  y = imageHeight + legendHeight * scale;
  for (const section of sections) {
    context.font = `${section.size * scale}px system-ui`; context.fillStyle = '#163540';
    for (const line of section.rows) { y += (section.size + 9) * scale; context.fillText(line, margin, y); }
    y += 12 * scale;
  }
  return canvas;
}
