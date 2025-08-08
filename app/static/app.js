const input = document.getElementById('inputText');
const button = document.getElementById('predictBtn');
const result = document.getElementById('result');

function setResult(text, cls) {
  result.textContent = text;
  result.className = `result ${cls || ''}`;
}

async function predict() {
  const text = (input.value || '').trim();
  if (!text) {
    setResult('Please enter some text first.', 'mutual');
    return;
  }
  setResult('Predicting…');
  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.error || 'Request failed');
    }
    const label = data.label || 'mutual';
    const pretty = label.charAt(0).toUpperCase() + label.slice(1);
    setResult(`Prediction: ${pretty}`, label);
  } catch (err) {
    setResult(`Error: ${err.message || err}`, 'bad');
  }
}

button.addEventListener('click', predict);
input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
    predict();
  }
});