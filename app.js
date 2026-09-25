/**
 * NN — Neural Network Engine & Interactive Visualizer
 * Pure JavaScript implementation of MLP Neural Network and interactive Canvas rendering.
 */

// ==========================================
// 1. Funções Matemáticas & Ativações
// ==========================================

const Activations = {
  sigmoid: {
    fn: (x) => 1 / (1 + Math.exp(-Math.max(-50, Math.min(50, x)))),
    df: (out) => out * (1 - out)
  },
  tanh: {
    fn: (x) => Math.tanh(x),
    df: (out) => 1 - out * out
  },
  relu: {
    fn: (x) => Math.max(0, x),
    df: (out) => (out > 0 ? 1 : 0)
  }
};

// ==========================================
// 2. Classe Rede Neural (MLP)
// ==========================================

class NeuralNetworkJS {
  constructor(topology, hiddenAct = 'tanh', outputAct = 'sigmoid') {
    this.topology = [...topology];
    this.hiddenActName = hiddenAct;
    this.outputActName = outputAct;
    this.initLayers();
  }

  initLayers() {
    this.layers = [];
    for (let l = 1; l < this.topology.length; l++) {
      const numIn = this.topology[l - 1];
      const numOut = this.topology[l];
      const isOutput = l === this.topology.length - 1;
      const act = isOutput ? Activations[this.outputActName] : Activations[this.hiddenActName];

      const neurons = [];
      const limit = Math.sqrt(2.0 / numIn);
      for (let n = 0; n < numOut; n++) {
        const weights = [];
        for (let w = 0; w < numIn; w++) {
          weights.push((Math.random() * 2 - 1) * limit);
        }
        neurons.push({
          weights: weights,
          bias: (Math.random() * 2 - 1) * 0.1,
          output: 0,
          delta: 0,
          act: act
        });
      }
      this.layers.push(neurons);
    }
  }

  forward(inputs) {
    let current = inputs;
    for (let l = 0; l < this.layers.length; l++) {
      const next = [];
      const layer = this.layers[l];
      for (let n = 0; n < layer.length; n++) {
        const neuron = layer[n];
        let sum = neuron.bias;
        for (let w = 0; w < current.length; w++) {
          sum += neuron.weights[w] * current[w];
        }
        neuron.output = neuron.act.fn(sum);
        next.push(neuron.output);
      }
      current = next;
    }
    return current;
  }

  backward(targets) {
    // Camada de saída
    const outputLayer = this.layers[this.layers.length - 1];
    for (let n = 0; n < outputLayer.length; n++) {
      const neuron = outputLayer[n];
      const error = targets[n] - neuron.output;
      neuron.delta = error * neuron.act.df(neuron.output);
    }

    // Camadas ocultas
    for (let l = this.layers.length - 2; l >= 0; l--) {
      const layer = this.layers[l];
      const nextLayer = this.layers[l + 1];
      for (let n = 0; n < layer.length; n++) {
        const neuron = layer[n];
        let error = 0;
        for (let nextN = 0; nextN < nextLayer.length; nextN++) {
          error += nextLayer[nextN].weights[n] * nextLayer[nextN].delta;
        }
        neuron.delta = error * neuron.act.df(neuron.output);
      }
    }
  }

  updateWeights(inputs, lr) {
    for (let l = 0; l < this.layers.length; l++) {
      const prevOutputs = l === 0 ? inputs : this.layers[l - 1].map((n) => n.output);
      const layer = this.layers[l];
      for (let n = 0; n < layer.length; n++) {
        const neuron = layer[n];
        for (let w = 0; w < neuron.weights.length; w++) {
          neuron.weights[w] += lr * neuron.delta * prevOutputs[w];
        }
        neuron.bias += lr * neuron.delta;
      }
    }
  }

  trainSample(inputs, targets, lr) {
    const outputs = this.forward(inputs);
    this.backward(targets);
    this.updateWeights(inputs, lr);
    let loss = 0;
    for (let i = 0; i < targets.length; i++) {
      loss += Math.pow(targets[i] - outputs[i], 2);
    }
    return loss / targets.length;
  }
}

// ==========================================
// 3. Geradores de Datasets 2D
// ==========================================

const Datasets = {
  xor: (numSamples = 160, noise = 0.05) => {
    const data = [];
    for (let i = 0; i < numSamples; i++) {
      const label = Math.random() > 0.5 ? 1 : 0;
      let x, y;
      if (label === 1) {
        if (Math.random() > 0.5) {
          x = 0.5 + (Math.random() - 0.5) * noise * 8;
          y = -0.5 + (Math.random() - 0.5) * noise * 8;
        } else {
          x = -0.5 + (Math.random() - 0.5) * noise * 8;
          y = 0.5 + (Math.random() - 0.5) * noise * 8;
        }
      } else {
        if (Math.random() > 0.5) {
          x = 0.5 + (Math.random() - 0.5) * noise * 8;
          y = 0.5 + (Math.random() - 0.5) * noise * 8;
        } else {
          x = -0.5 + (Math.random() - 0.5) * noise * 8;
          y = -0.5 + (Math.random() - 0.5) * noise * 8;
        }
      }
      data.push({ x: [x, y], y: [label] });
    }
    return data;
  },

  circle: (numSamples = 160, noise = 0.05) => {
    const data = [];
    for (let i = 0; i < numSamples; i++) {
      const label = Math.random() > 0.5 ? 1 : 0;
      const r = label === 0 ? Math.random() * 0.35 : 0.65 + Math.random() * 0.25;
      const angle = Math.random() * Math.PI * 2;
      const x = Math.cos(angle) * r + (Math.random() - 0.5) * noise;
      const y = Math.sin(angle) * r + (Math.random() - 0.5) * noise;
      data.push({ x: [x, y], y: [label] });
    }
    return data;
  },

  moons: (numSamples = 160, noise = 0.05) => {
    const data = [];
    const n = Math.floor(numSamples / 2);
    for (let i = 0; i < n; i++) {
      const theta = (Math.PI * i) / n;
      data.push({
        x: [Math.cos(theta) * 0.6 - 0.3 + (Math.random() - 0.5) * noise, Math.sin(theta) * 0.6 - 0.15 + (Math.random() - 0.5) * noise],
        y: [0]
      });
      data.push({
        x: [0.3 - Math.cos(theta) * 0.6 + (Math.random() - 0.5) * noise, 0.15 - Math.sin(theta) * 0.6 + (Math.random() - 0.5) * noise],
        y: [1]
      });
    }
    return data;
  },

  spiral: (numSamples = 160, noise = 0.05) => {
    const data = [];
    const n = Math.floor(numSamples / 2);
    for (let i = 0; i < n; i++) {
      const r = (i / n) * 0.85;
      const t = 1.75 * (i / n) * 2 * Math.PI;
      // Braço 0
      data.push({
        x: [r * Math.sin(t) + (Math.random() - 0.5) * noise, r * Math.cos(t) + (Math.random() - 0.5) * noise],
        y: [0]
      });
      // Braço 1
      data.push({
        x: [-r * Math.sin(t) + (Math.random() - 0.5) * noise, -r * Math.cos(t) + (Math.random() - 0.5) * noise],
        y: [1]
      });
    }
    return data;
  }
};

// ==========================================
// 4. Estado da Aplicação
// ==========================================

const state = {
  topology: [2, 4, 4, 1],
  datasetName: 'xor',
  learningRate: 0.15,
  hiddenActivation: 'tanh',
  noise: 0.05,
  isTraining: false,
  epoch: 0,
  lossHistory: [],
  dataset: []
};

let nn = new NeuralNetworkJS(state.topology, state.hiddenActivation, 'sigmoid');
state.dataset = Datasets[state.datasetName](160, state.noise);

// ==========================================
// 5. Elementos DOM
// ==========================================

const boundaryCanvas = document.getElementById('boundaryCanvas');
const boundaryCtx = boundaryCanvas.getContext('2d');

const networkCanvas = document.getElementById('networkCanvas');
const networkCtx = networkCanvas.getContext('2d');

const lossCanvas = document.getElementById('lossCanvas');
const lossCtx = lossCanvas.getContext('2d');

const btnToggleTrain = document.getElementById('btn-toggle-train');
const btnToggleText = document.getElementById('btn-toggle-text');
const btnStep = document.getElementById('btn-step');
const btnReset = document.getElementById('btn-reset');

const epochCountEl = document.getElementById('epoch-count');
const lossValueEl = document.getElementById('loss-value');
const lrSlider = document.getElementById('learning-rate');
const lrVal = document.getElementById('lr-val');
const noiseSlider = document.getElementById('noise-slider');
const noiseVal = document.getElementById('noise-val');
const actSelect = document.getElementById('activation-select');
const hiddenLayersTag = document.getElementById('hidden-layers-tag');

const btnAddLayer = document.getElementById('btn-add-layer');
const btnRemoveLayer = document.getElementById('btn-remove-layer');
const btnAddNeuron = document.getElementById('btn-add-neuron');
const btnRemoveNeuron = document.getElementById('btn-remove-neuron');

// ==========================================
// 6. Funções de Renderização dos Canvas
// ==========================================

// 6.1 Superfície de Decisão 2D
function renderBoundary() {
  const w = boundaryCanvas.width;
  const h = boundaryCanvas.height;
  const resolution = 30; // 30x30 grid para performance fluida
  const cellW = w / resolution;
  const cellH = h / resolution;

  for (let i = 0; i < resolution; i++) {
    for (let j = 0; j < resolution; j++) {
      // Normaliza de [-1, 1]
      const nx = (i / resolution) * 2 - 1;
      const ny = (j / resolution) * 2 - 1;

      const pred = nn.forward([nx, ny])[0];
      // Interpolação de cores: Classe 0 (Cyan: 6, 182, 212) -> Classe 1 (Magenta: 236, 72, 153)
      const r = Math.round(6 + (236 - 6) * pred);
      const g = Math.round(182 + (72 - 182) * pred);
      const b = Math.round(212 + (153 - 212) * pred);

      boundaryCtx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.22)`;
      boundaryCtx.fillRect(i * cellW, j * cellH, cellW + 1, cellH + 1);
    }
  }

  // Desenha os pontos do dataset
  for (const pt of state.dataset) {
    const px = ((pt.x[0] + 1) / 2) * w;
    const py = ((pt.x[1] + 1) / 2) * h;
    const isClassOne = pt.y[0] === 1;

    boundaryCtx.beginPath();
    boundaryCtx.arc(px, py, 4.5, 0, Math.PI * 2);
    boundaryCtx.fillStyle = isClassOne ? '#ec4899' : '#06b6d4';
    boundaryCtx.fill();
    boundaryCtx.lineWidth = 1.5;
    boundaryCtx.strokeStyle = '#ffffff';
    boundaryCtx.stroke();
  }
}

// 6.2 Topologia e Pesos da Rede Neural
function renderNetwork() {
  const w = networkCanvas.width;
  const h = networkCanvas.height;
  networkCtx.clearRect(0, 0, w, h);

  const numLayers = state.topology.length;
  const layerSpacing = (w - 100) / (numLayers - 1);
  const startX = 50;

  // Calcula posições (x, y) de cada neurônio
  const positions = [];
  for (let l = 0; l < numLayers; l++) {
    const numNeurons = state.topology[l];
    const layerPos = [];
    const neuronSpacing = (h - 80) / (numNeurons + 1);
    for (let n = 0; n < numNeurons; n++) {
      layerPos.push({
        x: startX + l * layerSpacing,
        y: 40 + (n + 1) * neuronSpacing
      });
    }
    positions.push(layerPos);
  }

  // 1. Desenha as conexões (pesos)
  for (let l = 1; l < numLayers; l++) {
    const currentLayer = nn.layers[l - 1];
    for (let n = 0; n < currentLayer.length; n++) {
      const neuron = currentLayer[n];
      const targetPos = positions[l][n];

      for (let wIdx = 0; wIdx < neuron.weights.length; wIdx++) {
        const sourcePos = positions[l - 1][wIdx];
        const weight = neuron.weights[wIdx];

        networkCtx.beginPath();
        networkCtx.moveTo(sourcePos.x, sourcePos.y);
        networkCtx.lineTo(targetPos.x, targetPos.y);

        const isPositive = weight >= 0;
        const alpha = Math.min(1, Math.max(0.15, Math.abs(weight) * 0.8));
        networkCtx.strokeStyle = isPositive
          ? `rgba(6, 182, 212, ${alpha})`
          : `rgba(236, 72, 153, ${alpha})`;
        networkCtx.lineWidth = Math.min(5, Math.max(1, Math.abs(weight) * 2));
        networkCtx.stroke();
      }
    }
  }

  // 2. Desenha os neurônios
  for (let l = 0; l < numLayers; l++) {
    const layerPos = positions[l];
    for (let n = 0; n < layerPos.length; n++) {
      const pos = layerPos[n];

      // Brilho exterior
      networkCtx.beginPath();
      networkCtx.arc(pos.x, pos.y, 14, 0, Math.PI * 2);
      networkCtx.fillStyle = 'rgba(99, 102, 241, 0.2)';
      networkCtx.fill();

      // Círculo principal
      networkCtx.beginPath();
      networkCtx.arc(pos.x, pos.y, 9, 0, Math.PI * 2);
      networkCtx.fillStyle = '#1e293b';
      networkCtx.fill();
      networkCtx.lineWidth = 2;
      networkCtx.strokeStyle = l === 0 ? '#06b6d4' : l === numLayers - 1 ? '#ec4899' : '#6366f1';
      networkCtx.stroke();
    }
  }

  // 3. Rótulos das camadas
  networkCtx.font = '11px Outfit, sans-serif';
  networkCtx.fillStyle = '#94a3b8';
  networkCtx.textAlign = 'center';
  for (let l = 0; l < numLayers; l++) {
    const x = startX + l * layerSpacing;
    const label = l === 0 ? 'Entrada' : l === numLayers - 1 ? 'Saída' : `Oculta ${l}`;
    networkCtx.fillText(label, x, h - 12);
  }
}

// 6.3 Gráfico de Loss
function renderLossChart() {
  const w = lossCanvas.width;
  const h = lossCanvas.height;
  lossCtx.clearRect(0, 0, w, h);

  if (state.lossHistory.length < 2) return;

  // Grid lines
  lossCtx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
  lossCtx.lineWidth = 1;
  for (let y = 0; y <= h; y += 30) {
    lossCtx.beginPath();
    lossCtx.moveTo(0, y);
    lossCtx.lineTo(w, y);
    lossCtx.stroke();
  }

  const maxPoints = 120;
  const history = state.lossHistory.slice(-maxPoints);
  const stepX = w / (maxPoints - 1);
  const maxLoss = 0.5; // Escala máxima

  // Área preenchida
  lossCtx.beginPath();
  for (let i = 0; i < history.length; i++) {
    const x = i * stepX;
    const normLoss = Math.min(1, history[i] / maxLoss);
    const y = h - normLoss * (h - 20) - 10;
    if (i === 0) lossCtx.moveTo(x, y);
    else lossCtx.lineTo(x, y);
  }
  lossCtx.lineTo((history.length - 1) * stepX, h);
  lossCtx.lineTo(0, h);
  lossCtx.closePath();

  const gradient = lossCtx.createLinearGradient(0, 0, 0, h);
  gradient.addColorStop(0, 'rgba(6, 182, 212, 0.35)');
  gradient.addColorStop(1, 'rgba(6, 182, 212, 0.0)');
  lossCtx.fillStyle = gradient;
  lossCtx.fill();

  // Linha superior
  lossCtx.beginPath();
  for (let i = 0; i < history.length; i++) {
    const x = i * stepX;
    const normLoss = Math.min(1, history[i] / maxLoss);
    const y = h - normLoss * (h - 20) - 10;
    if (i === 0) lossCtx.moveTo(x, y);
    else lossCtx.lineTo(x, y);
  }
  lossCtx.strokeStyle = '#06b6d4';
  lossCtx.lineWidth = 2.5;
  lossCtx.stroke();
}

// ==========================================
// 7. Loop de Treinamento
// ==========================================

function trainEpoch() {
  let totalLoss = 0;
  // Embaralha dataset
  const shuffled = [...state.dataset].sort(() => Math.random() - 0.5);

  for (let i = 0; i < shuffled.length; i++) {
    const sample = shuffled[i];
    const loss = nn.trainSample(sample.x, sample.y, state.learningRate);
    totalLoss += loss;
  }

  const avgLoss = totalLoss / shuffled.length;
  state.epoch++;
  state.lossHistory.push(avgLoss);

  // Atualiza indicadores UI
  epochCountEl.innerText = state.epoch.toLocaleString();
  lossValueEl.innerText = avgLoss.toFixed(4);

  renderBoundary();
  renderNetwork();
  renderLossChart();
}

function loop() {
  if (state.isTraining) {
    trainEpoch();
    requestAnimationFrame(loop);
  }
}

// ==========================================
// 8. Event Listeners & Controles
// ==========================================

btnToggleTrain.addEventListener('click', () => {
  state.isTraining = !state.isTraining;
  if (state.isTraining) {
    btnToggleTrain.classList.remove('btn-primary');
    btnToggleTrain.classList.add('btn-secondary');
    btnToggleText.innerText = 'Pausar';
    loop();
  } else {
    btnToggleTrain.classList.remove('btn-secondary');
    btnToggleTrain.classList.add('btn-primary');
    btnToggleText.innerText = 'Iniciar';
  }
});

btnStep.addEventListener('click', () => {
  if (!state.isTraining) {
    trainEpoch();
  }
});

function resetNetwork() {
  nn = new NeuralNetworkJS(state.topology, state.hiddenActivation, 'sigmoid');
  state.epoch = 0;
  state.lossHistory = [];
  epochCountEl.innerText = '0';
  lossValueEl.innerText = '0.0000';
  updateTopologyUI();
  renderBoundary();
  renderNetwork();
  renderLossChart();
}

btnReset.addEventListener('click', resetNetwork);

// Seletor de Dataset
document.querySelectorAll('.dataset-btn').forEach((btn) => {
  btn.addEventListener('click', (e) => {
    document.querySelectorAll('.dataset-btn').forEach((b) => b.classList.remove('active'));
    btn.classList.add('active');
    state.datasetName = btn.dataset.dataset;
    state.dataset = Datasets[state.datasetName](160, state.noise);
    resetNetwork();
  });
});

// Controles de Sliders
lrSlider.addEventListener('input', (e) => {
  state.learningRate = parseFloat(e.target.value);
  lrVal.innerText = state.learningRate.toFixed(2);
});

noiseSlider.addEventListener('input', (e) => {
  state.noise = parseFloat(e.target.value);
  noiseVal.innerText = state.noise.toFixed(2);
  state.dataset = Datasets[state.datasetName](160, state.noise);
  renderBoundary();
});

actSelect.addEventListener('change', (e) => {
  state.hiddenActivation = e.target.value;
  resetNetwork();
});

// Alteração Dinâmica de Arquitetura
function updateTopologyUI() {
  const hidden = state.topology.slice(1, -1);
  hiddenLayersTag.innerText = `Camadas: [${hidden.join(', ')}]`;
}

btnAddLayer.addEventListener('click', () => {
  if (state.topology.length < 6) {
    // Insere uma nova camada antes da saída com 4 neurônios
    state.topology.splice(state.topology.length - 1, 0, 4);
    resetNetwork();
  }
});

btnRemoveLayer.addEventListener('click', () => {
  if (state.topology.length > 3) {
    // Remove a última camada oculta
    state.topology.splice(state.topology.length - 2, 1);
    resetNetwork();
  }
});

btnAddNeuron.addEventListener('click', () => {
  // Adiciona +1 neurônio na primeira camada oculta (máx 8)
  if (state.topology[1] < 8) {
    state.topology[1]++;
    resetNetwork();
  }
});

btnRemoveNeuron.addEventListener('click', () => {
  // Remove 1 neurônio na primeira camada oculta (mín 2)
  if (state.topology[1] > 2) {
    state.topology[1]--;
    resetNetwork();
  }
});

// Inicialização inicial
updateTopologyUI();
renderBoundary();
renderNetwork();
renderLossChart();
