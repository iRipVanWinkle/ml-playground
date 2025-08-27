# End-to-End Testing with Playwright

This project uses [Playwright](https://playwright.dev/) for end-to-end testing of the ML Playground application.

## Setup

Playwright is already configured and set up for this project. The configuration can be found in `playwright.config.ts`.

## Running Tests

### Run all tests

```bash
npm run test:e2e
```

### Run tests with UI mode (visual test runner)

```bash
npm run test:e2e:ui
```

### Run tests in headed mode (see browser)

```bash
npm run test:e2e:headed
```

### Run tests in debug mode

```bash
npm run test:e2e:debug
```

### Run tests on specific browsers

```bash
# Chromium only
npx playwright test --project=chromium

# Firefox only
npx playwright test --project=firefox

# WebKit only
npx playwright test --project=webkit
```

## Test Structure

Tests are located in the `e2e/` directory:

- `e2e/ml-playground.spec.ts` - Main application tests covering:
    - Page loading and title verification
    - Tab navigation between Regression and Classification
    - Responsive layout testing
    - UI component visibility
    - Keyboard navigation
    - Console error detection

## Browser Support

Tests run against:

- **Chromium** (Chrome/Edge)
- **Firefox**
- **WebKit** (Safari)

## CI/CD Integration

Tests are automatically run in GitHub Actions on push and pull requests. See `.github/workflows/playwright.yml` for the CI configuration.

## Development

When running tests locally, Playwright will automatically:

1. Start the development server (`npm run dev`)
2. Wait for the app to be available at `http://localhost:5173`
3. Run the tests
4. Generate an HTML report

## Test Reports

After running tests, an HTML report is generated and automatically opened. You can also view it manually:

```bash
npx playwright show-report
```

## Writing New Tests

When adding new tests:

1. Create test files in the `e2e/` directory with `.spec.ts` extension
2. Use Playwright's built-in selectors and assertions
3. Follow the existing patterns for handling multiple matching elements
4. Test across different viewport sizes for responsive behavior

## Tips

- Use `page.getByRole()` for accessibility-friendly selectors
- Use `.first()` when multiple elements match to avoid strict mode violations
- Test both desktop and mobile viewports
- Use `page.waitForLoadState('domcontentloaded')` instead of `'networkidle'` for faster tests
