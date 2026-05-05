/**
 * Recursively walks `value` and collects every distinct ArrayBuffer underlying
 * a TypedArray or DataView, plus any standalone Transferable instance found
 * (ArrayBuffer, MessagePort, ImageBitmap, OffscreenCanvas).
 *
 * SharedArrayBuffer is intentionally excluded — it cannot be transferred.
 */
export function collectTransferables(value: unknown): Transferable[] {
    const buffers = new Set<Transferable>();
    const seen = new WeakSet<object>();

    const visit = (node: unknown) => {
        if (node === null || typeof node !== 'object') return;
        if (seen.has(node)) return;
        seen.add(node);

        if (ArrayBuffer.isView(node)) {
            const { buffer } = node as ArrayBufferView;
            if (buffer instanceof ArrayBuffer) buffers.add(buffer);
            return;
        }

        if (node instanceof ArrayBuffer) {
            buffers.add(node);
            return;
        }

        if (
            (typeof MessagePort !== 'undefined' && node instanceof MessagePort) ||
            (typeof ImageBitmap !== 'undefined' && node instanceof ImageBitmap) ||
            (typeof OffscreenCanvas !== 'undefined' && node instanceof OffscreenCanvas)
        ) {
            buffers.add(node as Transferable);
            return;
        }

        if (Array.isArray(node)) {
            for (const item of node) visit(item);
            return;
        }

        for (const key of Object.keys(node)) visit((node as Record<string, unknown>)[key]);
    };

    visit(value);
    return Array.from(buffers);
}
