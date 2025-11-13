/* Image preview and mock predictions */
(function () {
  const input = document.getElementById('imageInput');
  const preview = document.getElementById('preview');
  const predictBtn = document.getElementById('predictBtn');
  const help = document.getElementById('predictHelp');
  const result = document.getElementById('mockResult');
  const labelsEl = document.getElementById('classLabels');

  const LABELS = Array.isArray(window.__CLASS_LABELS__) ? window.__CLASS_LABELS__ : [];
  labelsEl.innerHTML = LABELS.map(l => `<span class="pill">${l}</span>`).join('');
  const DESC = {
    'Healthy': 'Your plant appears to be healthy. Continue regular care and monitoring.',
    'Early Blight': 'Early blight causes dark spots on leaves. Apply fungicide and remove affected leaves.',
    'Late Blight': 'Late blight is serious. Apply copper-based fungicide and improve air circulation.',
    'Bacterial Spot': 'Small dark spots indicate bacterial spot. Remove affected leaves and apply copper-based bactericide.'
  };

  input.addEventListener('change', (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const url = URL.createObjectURL(file);
    preview.innerHTML = `<img alt="Uploaded preview" src="${url}" />`;
    result.innerHTML = '';
  });

  predictBtn.addEventListener('click', () => {
    help.textContent = 'Demo / Preview only — not a real prediction.';
    const probs = randomDistribution(LABELS.length);
    const bestIdx = argmax(probs);
    const bestLabel = LABELS[bestIdx] || 'Unknown';
    const bestScore = probs[bestIdx];
    const statusClass = bestLabel === 'Healthy' ? 'status-healthy' : 'status-disease';
    const statusText = bestLabel === 'Healthy' ? 'HEALTHY' : 'DISEASED';
    const descText = DESC[bestLabel] || '';

    const rows = LABELS.map((l, i) => {
      const pct = (probs[i] * 100).toFixed(1);
      return `<div class="prob-row"><span>${l}</span><span>${pct}%</span></div>`;
    }).join('');

    const pct = (bestScore * 100).toFixed(1);
    result.innerHTML = `
      <span class="status-badge ${statusClass}">${statusText}</span>
      <div class="title">Predicted (mock): ${bestLabel}</div>
      <p class="confidence-text">Confidence: <strong>${pct}%</strong></p>
      <div class="progress"><div class="bar" style="width:${pct}%"></div></div>
      <div class="muted small">${descText}</div>
      <div class="muted small">To get live predictions, run the Streamlit app locally or deploy to Streamlit Cloud. See instructions below.</div>
      <div class="prob-table">${rows}</div>
    `;
  });

  function randomDistribution(n) {
    const xs = Array.from({ length: n }, () => Math.random() + 0.05);
    const sum = xs.reduce((a, b) => a + b, 0);
    return xs.map(x => x / sum);
  }

  function argmax(arr) {
    return arr.reduce((bestIdx, x, i, a) => x > a[bestIdx] ? i : bestIdx, 0);
  }
})();