// server-graph.js
const express = require('express');
const path = require('path');
const fs = require('fs');
const tf = require('@tensorflow/tfjs');
const wasmBackend = require('@tensorflow/tfjs-backend-wasm');

const app = express();
app.use(express.json());

const modelDir = path.resolve(__dirname, 'model');
app.use('/model', express.static(modelDir));

let model = null;   // GraphModel
let ready = false;
let inputName = null; // nombre del tensor de entrada
let outputName = null;

const PORT = process.env.PORT || 3000;

function wasmFileDirUrl() {
  const dist = path.join(__dirname, 'node_modules', '@tensorflow', 'tfjs-backend-wasm', 'dist');
  return 'file://' + dist.replace(/\\/g, '/') + '/';
}

app.listen(PORT, async () => {
  console.log(` API (graph) en http://localhost:${PORT}`);
  try {
    const wasmPath = wasmFileDirUrl();
    wasmBackend.setWasmPaths(wasmPath);
    console.log('WASM file path prefix:', wasmPath);

    await tf.setBackend('wasm');
    await tf.ready();
    console.log(' Backend:', tf.getBackend());

    const url = `http://localhost:${PORT}/model/model.json`;
    console.log('⬇  Cargando GraphModel desde:', url);
    model = await tf.loadGraphModel(url);

    // Detecta nombres de entrada/salida (coger el primero suele valer)
    inputName = model.inputs?.[0]?.name || null;
    outputName = model.outputs?.[0]?.name || null;
    console.log('inputName:', inputName, '| outputName:', outputName);

    // warm-up
    const X = tf.tensor2d([[0, 0]], [1, 2], 'float32');
    const out = await model.executeAsync ? await model.executeAsync({ [inputName]: X })
                                         : model.execute({ [inputName]: X });
    (out.dispose ? out.dispose() : Array.isArray(out) && out.forEach(t => t.dispose?.()));
    X.dispose();

    ready = true;
    console.log('GraphModel cargado');
  } catch (err) {
    console.error(' Error al iniciar:', err?.message || err);
    process.exit(1);
  }
});

app.get('/health', (_req, res) => {
  res.json({ ok: ready, backend: tf.getBackend?.(), modelLoaded: !!model, inputName, outputName });
});

app.post('/predict', async (req, res) => {
  try {
    if (!ready || !model) return res.status(503).json({ error: 'Modelo no cargado' });

    const { points, threshold = 0.5 } = req.body;
    if (points === undefined) return res.status(400).json({ error: 'Falta "points"' });

    const batch = Array.isArray(points[0]) ? points : [points];
    for (const p of batch) {
      if (!Array.isArray(p) || p.length !== 2 ||
          !Number.isFinite(p[0]) || !Number.isFinite(p[1])) {
        return res.status(400).json({ error: 'Cada punto debe ser [x,y] numérico' });
      }
    }

    const X = tf.tensor2d(batch, [batch.length, 2], 'float32');

    // Ejecuta según soporte (algunas builds soportan .executeAsync)
    let out = null;
    if (typeof model.executeAsync === 'function') {
      out = await model.executeAsync({ [inputName]: X });
    } else {
      out = model.execute({ [inputName]: X });
    }

    // Convierte salida a array
    let probs2d;
    if (Array.isArray(out)) {
      probs2d = await out[0].array();
      out.forEach(t => t.dispose?.());
    } else {
      probs2d = await out.array();
      out.dispose?.();
    }
    X.dispose();

    const probs = Array.isArray(probs2d[0]) ? probs2d.map(([p]) => p) : probs2d;
    const classes = probs.map(p => (p >= threshold ? 1 : 0));
    res.json({ n: batch.length, probs, classes, threshold, backend: tf.getBackend(), inputName, outputName });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Error interno' });
  }
});
