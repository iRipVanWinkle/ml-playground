import { useMemo } from 'react';
import type { TypedArray } from '@/app/shared/helpers';

export function useImageHeatmap(weights: TypedArray, gridSize: number, min: number, max: number) {
    const context = useCanvasImageData(gridSize);

    return useMemo(() => {
        if (!context) return '';

        const imageData = context.createImageData(gridSize, gridSize);
        const data = imageData.data;

        for (let i = 0; i < weights.length; i++) {
            const rgb = getWeightColorRGB(weights[i], min, max);
            const pixelIndex = i * 4;
            data[pixelIndex] = rgb.r;
            data[pixelIndex + 1] = rgb.g;
            data[pixelIndex + 2] = rgb.b;
            data[pixelIndex + 3] = 255;
        }

        context.putImageData(imageData, 0, 0);

        return context.canvas.toDataURL();
    }, [context, gridSize, weights, min, max]);
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

function getWeightColorRGB(
    weight: number,
    min: number,
    max: number,
): { r: number; g: number; b: number } {
    const range = max - min;
    if (range === 0) return { r: 128, g: 128, b: 128 };

    // Normalize to 0-1
    const normalized = (weight - min) / range;

    // Create a diverging color scale: blue (negative) -> white (zero) -> red (positive)
    const midpoint = -min / range; // Where zero falls in the normalized range

    if (normalized < midpoint) {
        // Blue to white (negative values)
        const t = midpoint > 0 ? normalized / midpoint : 0;
        // HSL(220, 80%, 30%) to HSL(220, 80%, 100%)
        return hslToRgb(220, 0.8, 0.3 + t * 0.7);
    } else {
        // White to red (positive values)
        const t = midpoint < 1 ? (normalized - midpoint) / (1 - midpoint) : 0;
        // HSL(0, 80%, 100%) to HSL(0, 80%, 30%)
        return hslToRgb(0, 0.8, 1 - t * 0.7);
    }
}

function hslToRgb(h: number, s: number, l: number): { r: number; g: number; b: number } {
    const c = (1 - Math.abs(2 * l - 1)) * s;
    const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
    const m = l - c / 2;

    let r = 0,
        g = 0,
        b = 0;
    if (h < 60) {
        r = c;
        g = x;
        b = 0;
    } else if (h < 120) {
        r = x;
        g = c;
        b = 0;
    } else if (h < 180) {
        r = 0;
        g = c;
        b = x;
    } else if (h < 240) {
        r = 0;
        g = x;
        b = c;
    } else if (h < 300) {
        r = x;
        g = 0;
        b = c;
    } else {
        r = c;
        g = 0;
        b = x;
    }

    return {
        r: Math.round((r + m) * 255),
        g: Math.round((g + m) * 255),
        b: Math.round((b + m) * 255),
    };
}
