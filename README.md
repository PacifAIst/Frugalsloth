# FrugalSloth v0.3.3 [[LIFE DEMO](https://pacifaist.github.io/Frugalsloth/)] BETA

**Universal Edge AI Trainer & Inference Engine** — 100% browser-native, zero backend, zero data leaves your device.

Train neural networks (MLP from scratch), fine-tune transformers via ONNX, and export production-ready models — all in your browser.

![FrugalSloth](images/1.png)
![FrugalSloth](images/2.png)
![FrugalSloth](images/3.png)
![FrugalSloth](images/4.png)
![FrugalSloth](images/5.png)

## Quick Start

### Option 1: Static Web ZIP (Fastest — Any Hosting)
Download `frugalsloth-v0.3.3-web.zip` from the releases page, unzip it, and open `index.html` in any modern browser. That's it. No server required.

Or deploy the `dist/` folder to:
- **Cloudflare Pages** — drag & drop
- **GitHub Pages** — push `dist/` to a repo
- **Vercel/Netlify** — connect repo
- **Any static host** — SFTP upload

### Option 2: Windows .exe (Electron)
```bash
# On Windows, after cloning:
npm install
npm run dist
# Output: release/FrugalSloth-v0.3.3-Portable.exe
```

### Option 3: Development Server
```bash
npm install
npm run dev
# Open http://localhost:5173
```

## Features

| Feature | Description |
|---------|-------------|
| MLP Training | Multi-layer perceptron from scratch with TF.js WebGL |
| ONNX Export | Export trained models to ONNX format |
| Transformer Fine-tuning | Import ONNX models (BERT, etc.), train classifier head |
| 100% Private | No data ever leaves your browser |
| Offline Capable | Works without internet after load |
| Static Deploy | Single ZIP drops into any host |

## Architecture

```
Browser (Your Device)
├── TF.js WebGL Backend — training + inference
├── ONNX Runtime Web — transformer inference + export
├── IndexedDB — model persistence
├── Web Workers — background inference
└── Vite + React + TypeScript — UI
```

No server. No API keys. No subscription.

## File Structure

```
frugalsloth/
├── electron/
│   └── main.cjs              # Electron main process
├── public/
│   ├── icon.png              # App icon
│   ├── coi-serviceworker.js  # Cross-origin isolation for WASM threads
│   └── ort-*.wasm / ort-*.mjs # ONNX Runtime Web binaries
├── src/
│   ├── App.tsx               # Main app (all tabs)
│   ├── terminal.css          # Dual-theme design system
│   ├── types/frugalsloth.ts  # TypeScript interfaces
│   ├── utils/
│   │   ├── csvParser.ts      # CSV/JSON/JSONL parsing
│   │   ├── onnxExporter.ts   # ONNX model export
│   │   ├── engineTemplate.ts # Standalone inference engine generator
│   │   └── indexedDB.ts      # Model persistence
│   └── workers/
│       ├── inference.worker.ts   # Background inference
│       └── transformer.worker.ts # Transformer pipeline
├── dist/                     # Built static files (npm run build)
├── release/                  # Electron builds (npm run dist)
└── package.json              # Scripts + electron-builder config
```

## Build Scripts

| Script | Output |
|--------|--------|
| `npm run build` | Static files in `dist/` |
| `npm run zip` | `frugalsloth-v0.3.3-web.zip` for hosting |
| `npm run electron` | Launch Electron dev mode |
| `npm run dist` | Windows `.exe` (portable) |
| `npm run electron:build:win` | Windows installer + portable |
| `npm run electron:build:mac` | macOS `.dmg` |
| `npm run electron:build:linux` | Linux `.AppImage` |

## Dataset Formats

**CSV:** First row = headers. Target column = last column by default.
```csv
feature1,feature2,feature3,label
1.0,2.0,3.0,classA
4.0,5.0,6.0,classB
```

**JSON:** Array of objects with consistent keys.
```json
[
  {"feature1": 1.0, "feature2": 2.0, "label": "classA"},
  {"feature1": 4.0, "feature2": 5.0, "label": "classB"}
]
```

**JSONL:** One JSON object per line.
```jsonl
{"feature1": 1.0, "feature2": 2.0, "label": "classA"}
{"feature1": 4.0, "feature2": 5.0, "label": "classB"}
```

## Model Export Formats

| Format | Use Case |
|--------|----------|
| JSON Weights | Reload in FrugalSloth, use in custom JS code |
| ONNX | Deploy to ONNX Runtime (Python, C++, mobile) |
| Engine JS | Self-contained `<script>` tag for any HTML page |

## Version History

- **v0.3.3** — Train tab empty state, Electron setup, static ZIP
- **v0.3.2** — Cache busting, hard flush, colored buttons, tooltips
- **v0.3.0** — Model versioning, auto-naming (name_v1, name_v2)
- **v0.2.0** — Full UI overhaul, dual theme, docs tab
- **v0.1.0** — Initial release, MLP training, ONNX export

## License

[GNU AFFERO GENERAL PUBLIC LICENSE V3](https://github.com/PacifAIst/Frugalsloth/blob/main/LICENSE)
