# KneeXpert

Clinical web app for knee osteoarthritis workflows: patient management, X-ray and MRI diagnostics, batch cohort review, and report generation. The UI talks to the separate backbone API for AI inference.

## Prerequisites

- **Node.js** 18 or newer (20 LTS recommended)
- **npm** 9+ (bundled with Node)

## Setup

1. Open a terminal in the `KneeXpert` directory (this folder).

2. Install dependencies:

   ```bash
   npm install
   ```

3. Configure environment variables:

   ```bash
   cp .env.example .env
   ```

   Edit `.env` if your API is not on the default host:

   ```env
   VITE_BACKBONE_URL=http://localhost:9000
   ```

   The diagnostics workspace, batch analysis, and report assets require a running backbone service at this URL. See the backbone project README for how to start that server.

4. Start the development server:

   ```bash
   npm run dev
   ```

   Vite prints the local URL (typically `http://localhost:5173`). Open it in your browser.

## Scripts

| Command | Description |
| --- | --- |
| `npm run dev` | Start dev server with hot reload |
| `npm run build` | Production build to `dist/` |
| `npm run preview` | Serve the production build locally |
| `npm run lint` | Run ESLint |
| `npm test` | Run unit tests (Vitest) |

## Production build

```bash
npm run build
```

Deploy the contents of `dist/`. Set `VITE_BACKBONE_URL` at build time to the production backbone URL (Vite inlines env vars when you build).

## Troubleshooting

- **Diagnostics fail or stay offline** — Confirm the backbone API is up and that `VITE_BACKBONE_URL` in `.env` matches it. Restart `npm run dev` after changing `.env`.
- **Port already in use** — Run `npm run dev -- --port 5174` (or another free port).
- **Stale install** — Remove `node_modules` and run `npm install` again.
