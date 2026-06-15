# KneeXpert

Knee osteoarthritis diagnostics platform combining a **FastAPI ML backend** (X-ray KL-grade ensemble + Grad-CAM, and MRI MACS-Net + DeiT-S pipeline) with a **Vite + React + TypeScript** clinical UI.

```
KneeXpert/
├── backend/      FastAPI ML API (Python)        → http://localhost:9000
├── frontend/     Vite + React + shadcn UI       → http://localhost:8080
├── docker-compose.yml
└── README.md
```

The two pieces talk over a single HTTP boundary: the frontend reads `VITE_BACKBONE_URL` and calls the backend's REST endpoints (see `frontend/src/lib/diagnosticApi.ts`). They run as **two processes** — combining the repo gives you one place to clone and one command to launch both.

---

## Model weights (required, not in git)

Model checkpoints are large and **excluded from version control**. Place them locally before running:

- X-ray weights → `backend/xray/models/` (filenames in `backend/xray/loader.py`)
- MRI checkpoints → `backend/mri/models/` (filenames in `backend/mri/config.py`)
- Optional dev MRI sample → `backend/data/samples/`

Verify they're detected at `GET http://localhost:9000/health`.

---

## Quick start — Docker (recommended)

Runs backend + frontend together:

```bash
cd KneeXpert
docker compose up --build
```

- UI: http://localhost:8080
- API: http://localhost:9000/health

Weights and data are mounted from your local `backend/` folders, not baked into the image.

---

## Manual setup

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 9000 --reload
```

### Frontend

```bash
cd frontend
npm install                      # or: bun install
npm run dev                      # serves on http://localhost:8080
```

The frontend's `.env` sets `VITE_BACKBONE_URL=http://localhost:9000`. Adjust if the backend runs elsewhere.

---

## Backend API endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET  | `/health` | Service status + available models |
| POST | `/api/xray/predict` | Single X-ray image — `file`, optional `model_names` |
| POST | `/api/mri/predict` | Single MRI volume |
| POST | `/api/mri/predict/sample` | Dev sample volume (no upload) |
| GET  | `/api/mri/categories` | MRI label categories |
| POST | `/predict` | Batch X-ray images |

See `backend/README.md` for the full model catalog and details.
