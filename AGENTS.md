# ML Playground — Agent Instructions

## Project Overview

Interactive web app for exploring machine learning in the browser. Built with React 19, TypeScript, TensorFlow.js, and Vite.

## Architecture

- **App Layer** (`src/app/`): React components, UI sections, Zustand state management
- **ML Layer** (`src/ml/`): Pure ML algorithm implementations (independent of UI)
- **Web Workers**: Training runs in background threads
- **UI**: Radix UI primitives + shadcn/ui + Tailwind CSS v4

## Setup

```bash
nvm use           # Node 25 (see .nvmrc)
npm ci            # Install dependencies
```

## Key Commands

| Command              | Purpose                                    |
| -------------------- | ------------------------------------------ |
| `npm run dev`        | Start dev server (http://localhost:5173)    |
| `npm run build`      | Production build (typecheck + vite build)   |
| `npm run lint`       | ESLint check (.ts, .tsx files)             |
| `npm run lint:fix`   | ESLint auto-fix                            |
| `npm run format`     | Prettier format all files                  |
| `npm run typecheck`  | TypeScript type check (`tsc --noEmit`)     |
| `npm run test`       | Run unit tests (Vitest)                    |
| `npm run test:e2e`   | Run E2E tests (Playwright)                |
| `npm run check`      | Run all checks: lint, format, typecheck, test, e2e |

## Validation

Before submitting changes, always run:

```bash
npm run check
```

This runs lint, format check, typecheck, unit tests, and e2e tests.

## Code Style

- TypeScript strict mode
- ESLint + Prettier enforced
- Path alias: `@/*` maps to `./src/*`
- Inside `src/ml/` use relative imports (`../`, `./`) instead of `@/ml/...` — the ML layer must stay independent of the app alias
- Conventional commit messages
- Tests live next to source files (Vitest) or in `e2e/` (Playwright)

## Testing

- **Unit tests**: Vitest with jsdom environment, config in `vitest.config.ts`
- **E2E tests**: Playwright (Chromium, Firefox, WebKit), config in `playwright.config.ts`
- On CI, Playwright builds the app first then serves via `vite preview`

## Important Notes

- WASM files for TensorFlow.js are copied to `public/wasm/` via `npm run copy-wasm` (runs automatically before dev/build)
- The `global: 'globalThis'` define in vite config is required for plotly.js compatibility
- Production base path is `/ml-playground/` for GitHub Pages deployment
