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

    const rows = LABELS.map((l, i) => {
      const pct = (probs[i] * 100).toFixed(1);
      return `<div class="prob-row"><span>${l}</span><span>${pct}%</span></div>`;
    }).join('');

    result.innerHTML = `
      <div class="title">Predicted (mock): ${bestLabel} — ${(bestScore*100).toFixed(1)}%</div>
      <div class="muted small">To get live predictions, run the Streamlit app locally or deploy it to Streamlit Cloud. See instructions below.</div>
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