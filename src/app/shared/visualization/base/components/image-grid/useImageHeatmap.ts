import type { TypedArray } from '@/app/shared/helpers';
import { useMemo } from 'react';
import { useColor } from '../../../colors';

type RGB = { r: number; g: number; b: number };
type Colors = { dark: RGB; base: RGB; light: RGB };
type ColorSchema = {
    neg: Colors;
    diag: Colors;
    pos: Colors;
    zero: Colors;
};

type ImageHeatmapParams = {
    values: TypedArray;
    gridSize: number;
    min: number;
    max: number;
    showDiagonal?: boolean;
};

export function useImageHeatmap({
    values,
    gridSize,
    min,
    max,
    showDiagonal = false,
}: ImageHeatmapParams) {
    const context = useCanvasImageData(gridSize);
    const { getColor } = useColor();

    const colorSchema = useMemo(
        () => ({
            neg: {
                dark: hexToRgb(getColor('blue', 'darken')),
                base: hexToRgb(getColor('blue', 'base')),
                light: hexToRgb(getColor('blue', 'lighten')),
            },
            diag: {
                dark: hexToRgb(getColor('green', 'darken')),
                base: hexToRgb(getColor('green', 'base')),
                light: hexToRgb(getColor('green', 'lighten')),
            },
            pos: {
                dark: hexToRgb(getColor('red', 'darken')),
                base: hexToRgb(getColor('red', 'base')),
                light: hexToRgb(getColor('red', 'lighten')),
            },
            zero: {
                dark: hexToRgb(getColor('zero', 'darken')),
                base: hexToRgb(getColor('zero', 'base')),
                light: hexToRgb(getColor('zero', 'lighten')),
            },
        }),
        [getColor],
    );

    return useMemo(() => {
        if (!context) return '';

        const imageData = context.createImageData(gridSize, gridSize);
        const data = imageData.data;

        for (let i = 0; i < values.length && i < gridSize * gridSize; i++) {
            const row = Math.floor(i / gridSize);
            const col = i % gridSize;
            const showDiagonalElement = showDiagonal && row === col;

            const rgb = getValueColorRGB(values[i], min, max, showDiagonalElement, colorSchema);
            const pixelIndex = i * 4;
            data[pixelIndex] = rgb.r;
            data[pixelIndex + 1] = rgb.g;
            data[pixelIndex + 2] = rgb.b;
            data[pixelIndex + 3] = 255;
        }

        context.putImageData(imageData, 0, 0);

        return context.canvas.toDataURL();
    }, [context, gridSize, values, min, max, showDiagonal, colorSchema]);
}

function useCanvasImageData(gridSize: number) {
    return useMemo(() => {
        const canvas = document.createElement('canvas');
        canvas.width = gridSize;
        canvas.height = gridSize;
        const context = canvas.getContext('2d');

        if (!context) return null;

        return context;
    }, [gridSize]);
}

function getValueColorRGB(
    value: number,
    min: number,
    max: number,
    isDiagonal: boolean,
    colorSchema: ColorSchema,
): RGB {
    const t = (value - min) / (max - min);
    const zero = -min / (max - min);

    // Diagonal → green scale
    if (isDiagonal) {
        return lerpRGB(colorSchema.zero.light, colorSchema.diag.dark, clamp(t));
    }

    // Negative → blue → white
    if (t <= zero) {
        const k = clamp(t / zero);
        return lerpRGB(colorSchema.neg.dark, colorSchema.zero.light, k);
    }

    // Positive → white → red
    const k = clamp((t - zero) / (1 - zero));
    return lerpRGB(colorSchema.zero.light, colorSchema.pos.dark, k);
}

function clamp(v: number) {
    return Math.max(0, Math.min(1, v));
}

function lerp(a: number, b: number, t: number) {
    return a + (b - a) * t;
}

const lerpRGB = (a: RGB, b: RGB, t: number): RGB => ({
    r: (lerp(a.r, b.r, t) + 0.5) | 0,
    g: (lerp(a.g, b.g, t) + 0.5) | 0,
    b: (lerp(a.b, b.b, t) + 0.5) | 0,
});

function hexToRgb(hex: string): RGB {
    const n = parseInt(hex.slice(1), 16);
    return { r: (n >> 16) & 255, g: (n >> 8) & 255, b: n & 255 };
}
