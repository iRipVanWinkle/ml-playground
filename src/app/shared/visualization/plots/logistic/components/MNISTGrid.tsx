import { useCallback, useEffect, useMemo, useRef } from 'react';
import { usePlotlyColors } from '../../../colors';
import { useMNISTGridFrame } from '../hooks';

interface MNISTGridProps {
    data: number[][];
    predictions?: number[];
    labels: number[];
    originalLabels: string[];
}

type DigitColor = [number, number, number];

interface MNISTDigitCanvasProps {
    frame: number[][];
    color: DigitColor;
    bgColor: DigitColor;
    label: string;
    labelColor: string;
}

function MNISTDigitCanvas({ frame, color, bgColor, label, labelColor }: MNISTDigitCanvasProps) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const size = frame.length;

    const drawDigit = useCallback(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const imageData = ctx.createImageData(size, size);
        const { data } = imageData;

        // Normalize by the max value in the frame to use the full color range
        const maxVal = Math.max(...frame.flat());
        const scale = maxVal || 1;

        const [r, g, b] = color;
        const [bgR, bgG, bgB] = bgColor;

        for (let row = 0; row < size; row++) {
            for (let col = 0; col < size; col++) {
                const pixelIndex = (row * size + col) * 4;
                // Lerp from background color to target color based on intensity
                const t = frame[row][col] / scale;
                data[pixelIndex] = bgR + (r - bgR) * t; // R
                data[pixelIndex + 1] = bgG + (g - bgG) * t; // G
                data[pixelIndex + 2] = bgB + (b - bgB) * t; // B
                data[pixelIndex + 3] = 255; // A
            }
        }

        ctx.putImageData(imageData, 0, 0);
    }, [frame, color, bgColor, size]);

    useEffect(() => {
        drawDigit();
    }, [drawDigit]);

    return (
        <div className="flex flex-col items-center gap-1">
            <canvas
                ref={canvasRef}
                width={size}
                height={size}
                className="w-full aspect-square"
                style={{ imageRendering: 'pixelated' }}
            />
            <span className="text-xs text-center leading-tight" style={{ color: labelColor }}>
                {label}
            </span>
        </div>
    );
}

export function MNISTGrid({ data, labels, originalLabels, predictions }: MNISTGridProps) {
    const frames = useMNISTGridFrame(data);
    const plotlyColors = usePlotlyColors();

    const digitColors = useMemo(
        () => ({
            default: [0, 0, 0] as DigitColor,
            correct: [0, 150, 0] as DigitColor,
            incorrect: [200, 0, 0] as DigitColor,
        }),
        [],
    );

    const bgColor: DigitColor = [255, 255, 255];

    return (
        <div className="grid grid-cols-6 gap-4 w-full aspect-square pt-2.5 pr-5 pb-5 pl-2.5">
            {frames.map((frame, index) => {
                const digitLabel = labels[index];
                const predictedLabel = predictions?.[index];
                const color = predictions
                    ? predictedLabel === digitLabel
                        ? digitColors.correct
                        : digitColors.incorrect
                    : digitColors.default;
                const label =
                    originalLabels[digitLabel] +
                    (predictions ? ` (${originalLabels[predictions[index]]})` : '');

                return (
                    <MNISTDigitCanvas
                        key={index}
                        frame={frame}
                        color={color}
                        bgColor={bgColor}
                        label={label}
                        labelColor={plotlyColors.textColor}
                    />
                );
            })}
        </div>
    );
}
