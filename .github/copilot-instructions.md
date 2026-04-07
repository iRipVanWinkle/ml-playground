# ML Playground — Copilot Instructions

## Project Overview

Interactive web app for exploring machine learning in the browser. Built with React 19, TypeScript, TensorFlow.js, and Vite.

## Architecture

- **App Layer** (`src/app/`): React components, UI sections, Zustand state management
- **ML Layer** (`src/ml/`): Pure ML algorithm implementations (independent of UI)
- **Web Workers**: Training runs in background threads
- **UI**: Radix UI primitives + shadcn/ui + Tailwind CSS v4

## Code Style

- TypeScript strict mode
- ESLint + Prettier enforced (see `eslint.config.js`, `.prettierrc.json`)
- Path alias: `@/*` maps to `./src/*`
- Inside `src/ml/` use relative imports (`../`, `./`) instead of `@/ml/...` — the ML layer must stay independent of the app alias
- Conventional commit messages
- Do not add `React` imports — React 19 JSX transform handles it

## Testing

- **Unit tests**: Vitest with jsdom, co-located with source files
- **E2E tests**: Playwright (Chromium, Firefox, WebKit) in `e2e/`
- Write tests for new features; maintain existing test coverage

## Important Patterns

- State management uses Zustand with actions pattern
- ML algorithms are pure TypeScript — no React dependencies in `src/ml/`
- TensorFlow.js is used for tensor operations; most algorithms are implemented from scratch
- WASM files for TF.js are copied to `public/wasm/` automatically during build
