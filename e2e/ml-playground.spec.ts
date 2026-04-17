import { test, expect } from '@playwright/test';

test.describe('ML Playground', () => {
    test('should display main page elements and allow switching between regression and classification', async ({
        page,
    }) => {
        // Navigate to the main page
        await page.goto('/');

        // Check if the page loads with correct title
        await expect(page).toHaveTitle('Machine Learning Playground');

        // Check if main tabs are visible
        await expect(page.getByRole('tab', { name: 'Regression' })).toBeVisible();
        await expect(page.getByRole('tab', { name: 'Classification' })).toBeVisible();

        // Check if main sections are present using first() to handle multiple matches
        await expect(page.getByText('Dataset').first()).toBeVisible();
        await expect(page.getByText('Model').first()).toBeVisible();

        // Verify that Regression tab is selected by default
        await expect(page.getByRole('tab', { name: 'Regression' })).toHaveAttribute(
            'data-state',
            'active',
        );

        // Switch to Classification tab
        await page.getByRole('tab', { name: 'Classification' }).click();
        await expect(page.getByRole('tab', { name: 'Classification' })).toHaveAttribute(
            'data-state',
            'active',
        );

        // Switch back to Regression tab
        await page.getByRole('tab', { name: 'Regression' }).click();
        await expect(page.getByRole('tab', { name: 'Regression' })).toHaveAttribute(
            'data-state',
            'active',
        );
    });

    test('should handle tab navigation with keyboard', async ({ page }) => {
        await page.goto('/');

        // Focus on regression tab
        await page.getByRole('tab', { name: 'Regression' }).focus();

        // Use arrow keys to navigate between tabs
        await page.keyboard.press('ArrowRight');
        await expect(page.getByRole('tab', { name: 'Classification' })).toBeFocused();

        await page.keyboard.press('ArrowLeft');
        await expect(page.getByRole('tab', { name: 'Regression' })).toBeFocused();

        // Use Enter or Space to activate tab
        await page.keyboard.press('ArrowRight');
        await page.keyboard.press('Enter');
        await expect(page.getByRole('tab', { name: 'Classification' })).toHaveAttribute(
            'data-state',
            'active',
        );
    });

    test('should not have any console errors on load', async ({ page }) => {
        const consoleMessages: string[] = [];

        page.on('console', (msg) => {
            if (msg.type() === 'error') {
                consoleMessages.push(msg.text());
            }
        });

        await page.goto('/');

        // Wait a bit for any potential errors to surface
        await page.waitForTimeout(2000);

        // Check that no console errors occurred
        expect(consoleMessages).toHaveLength(0);
    });
});
