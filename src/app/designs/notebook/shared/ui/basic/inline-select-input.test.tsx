import * as React from 'react';
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { describe, expect, it } from 'vitest';
import { InlineSelectInput } from './inline-select-input';

(globalThis as unknown as { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

function render(element: React.ReactElement) {
    const container = document.createElement('div');
    document.body.appendChild(container);
    const root = createRoot(container);
    act(() => {
        root.render(element);
    });
    return {
        container,
        unmount: () => {
            act(() => {
                root.unmount();
            });
            container.remove();
        },
    };
}

describe('InlineSelectInput', () => {
    it('exposes compound subcomponents including input toggler controls', () => {
        expect(InlineSelectInput.Trigger).toBeDefined();
        expect(InlineSelectInput.Content).toBeDefined();
        expect(InlineSelectInput.Label).toBeDefined();
        expect(InlineSelectInput.Input).toBeDefined();
        expect(InlineSelectInput.Range).toBeDefined();
        expect(InlineSelectInput.Slider).toBe(InlineSelectInput.Range);
        expect(InlineSelectInput.Toggle).toBeDefined();
        expect(InlineSelectInput.Toggler).toBe(InlineSelectInput.Toggle);
        expect(InlineSelectInput.Switch).toBe(InlineSelectInput.Toggle);
        expect(InlineSelectInput.Hint).toBeDefined();
        expect(InlineSelectInput.Footer).toBeDefined();
        expect(InlineSelectInput.Done).toBeDefined();
        expect(InlineSelectInput.Close).toBe(InlineSelectInput.Done);
    });

    it('renders Toggle with switch variant and handles default label', () => {
        const { unmount } = render(
            <InlineSelectInput value={true} defaultOpen={true}>
                <InlineSelectInput.Content>
                    <InlineSelectInput.Toggle />
                </InlineSelectInput.Content>
            </InlineSelectInput>,
        );
        expect(document.body.textContent).toContain('Enabled');
        unmount();
    });

    it('renders Toggle with custom onLabel and offLabel', () => {
        const { unmount } = render(
            <InlineSelectInput value={false} defaultOpen={true}>
                <InlineSelectInput.Content>
                    <InlineSelectInput.Toggle onLabel="Shuffled" offLabel="Unshuffled" />
                </InlineSelectInput.Content>
            </InlineSelectInput>,
        );
        expect(document.body.textContent).toContain('Unshuffled');
        unmount();
    });

    it('renders Toggle with pills variant', () => {
        const { unmount } = render(
            <InlineSelectInput value="on" defaultOpen={true}>
                <InlineSelectInput.Content>
                    <InlineSelectInput.Toggle
                        variant="pills"
                        onLabel="Shuffled"
                        offLabel="Unshuffled"
                    />
                </InlineSelectInput.Content>
            </InlineSelectInput>,
        );
        expect(document.body.textContent).toContain('Shuffled');
        expect(document.body.textContent).toContain('Unshuffled');
        unmount();
    });
});
