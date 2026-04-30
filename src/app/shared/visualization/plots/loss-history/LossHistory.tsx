import { useLayoutEffect, useMemo, useRef, useState } from 'react';
import { line as d3line, scaleLinear } from 'd3';
import type { TrainingReport } from '@/app/models/types';
import type { Dataset } from '@/app/shared/types';
import { useColor } from '../../colors';

type LossHistoryProps = {
    dataset: Dataset;
    report: TrainingReport;
    testLossHistory?: number[][];
};

const PAD = { top: 16, right: 20, bottom: 30, left: 50 };
const Y_TICKS = 4;
const FALLBACK_SIZE = { width: 820, height: 260 };

export function LossHistory({ dataset, report, testLossHistory }: LossHistoryProps) {
    const { getColor } = useColor();
    const containerRef = useRef<HTMLDivElement>(null);
    const [size, setSize] = useState(FALLBACK_SIZE);

    const trainLossHistory =
        'trainLossHistory' in report ? report.trainLossHistory : undefined;

    useLayoutEffect(() => {
        const el = containerRef.current;
        if (!el) return;
        const ro = new ResizeObserver((entries) => {
            const entry = entries[0];
            if (!entry) return;
            const { width, height } = entry.contentRect;
            if (width > 0 && height > 0) setSize({ width, height });
        });
        ro.observe(el);
        return () => ro.disconnect();
    }, []);

    const chart = useMemo(() => {
        if (!trainLossHistory?.length) return null;

        const train = trainLossHistory;
        const test = testLossHistory ?? [];
        const maxLen = Math.max(
            1,
            ...train.map((s) => s.length),
            ...test.map((s) => s.length),
        );
        const allValues = [...train.flat(), ...test.flat()];
        const yMaxRaw = allValues.length ? Math.max(...allValues) : 1;
        const yMax = Math.max(1, yMaxRaw * 1.1);

        const innerW = Math.max(1, size.width - PAD.left - PAD.right);
        const innerH = Math.max(1, size.height - PAD.top - PAD.bottom);

        const xScale = scaleLinear()
            .domain([0, Math.max(1, maxLen - 1)])
            .range([PAD.left, PAD.left + innerW]);
        const yScale = scaleLinear()
            .domain([0, yMax])
            .range([PAD.top + innerH, PAD.top]);

        const buildPath = d3line<number>()
            .x((_, i) => xScale(i))
            .y((v) => yScale(v));

        const yTicks = Array.from({ length: Y_TICKS + 1 }, (_, i) => ({
            value: yMax - (i / Y_TICKS) * yMax,
            y: PAD.top + (i / Y_TICKS) * innerH,
        }));

        return { train, test, maxLen, xScale, yScale, buildPath, yTicks, innerW, innerH };
    }, [trainLossHistory, testLossHistory, size.width, size.height]);

    if (!trainLossHistory?.length || !chart) return null;

    const { train, test, maxLen, xScale, yScale, buildPath, yTicks, innerW, innerH } = chart;
    const showTest = test.length > 0;
    const categories = dataset.categories;
    const showClassLegend = train.length > 1 || Boolean(categories?.length);

    return (
        <div className="flex h-80 w-full flex-col">
            <div
                ref={containerRef}
                className="bg-muted/40 min-h-[200px] flex-1 rounded-2xl px-4 py-3"
            >
                <svg
                    viewBox={`0 0 ${size.width} ${size.height}`}
                    preserveAspectRatio="none"
                    className="text-muted-foreground block h-full w-full font-mono text-[10px]"
                    role="img"
                    aria-label="Loss curve"
                >
                    {yTicks.map((t, i) => (
                        <g key={`grid-${i}`}>
                            <line
                                x1={PAD.left}
                                x2={PAD.left + innerW}
                                y1={t.y}
                                y2={t.y}
                                className="stroke-border"
                                strokeDasharray="2 3"
                                opacity={0.6}
                            />
                            <text
                                x={PAD.left - 6}
                                y={t.y + 3}
                                textAnchor="end"
                                fill="currentColor"
                            >
                                {t.value.toFixed(2)}
                            </text>
                        </g>
                    ))}

                    <line
                        x1={PAD.left}
                        x2={PAD.left + innerW}
                        y1={PAD.top + innerH}
                        y2={PAD.top + innerH}
                        className="stroke-border"
                    />

                    <text x={PAD.left} y={size.height - 8} fill="currentColor">
                        0
                    </text>
                    <text
                        x={PAD.left + innerW}
                        y={size.height - 8}
                        textAnchor="end"
                        fill="currentColor"
                    >
                        {maxLen}
                    </text>

                    {showTest &&
                        test.map((vals, idx) => {
                            const path = buildPath(vals);
                            if (!path) return null;
                            return (
                                <path
                                    key={`test-${idx}`}
                                    d={path}
                                    fill="none"
                                    stroke={getColor(idx)}
                                    strokeWidth={1.5}
                                    strokeDasharray="3 3"
                                    opacity={0.85}
                                />
                            );
                        })}

                    {train.map((vals, idx) => {
                        const path = buildPath(vals);
                        if (!path) return null;
                        return (
                            <path
                                key={`train-${idx}`}
                                d={path}
                                fill="none"
                                stroke={getColor(idx)}
                                strokeWidth={1.75}
                            />
                        );
                    })}

                    {train.map((vals, idx) => {
                        if (!vals.length) return null;
                        const lastI = vals.length - 1;
                        return (
                            <circle
                                key={`dot-${idx}`}
                                cx={xScale(lastI)}
                                cy={yScale(vals[lastI])}
                                r={3}
                                className="fill-background"
                                stroke={getColor(idx)}
                                strokeWidth={1.5}
                            />
                        );
                    })}
                </svg>
            </div>

            <div className="text-muted-foreground mt-3 flex flex-wrap items-center justify-between gap-x-4 gap-y-2 font-mono text-[11px] tracking-wider uppercase">
                <span className="text-foreground font-bold">FIG. 01 — Loss curve</span>
                <span className="flex flex-wrap items-center gap-x-4 gap-y-1">
                    {showClassLegend &&
                        train.map((_, idx) => (
                            <span
                                key={`legend-class-${idx}`}
                                className="inline-flex items-center gap-1.5 normal-case tracking-normal"
                            >
                                <span
                                    aria-hidden
                                    className="inline-block h-[2px] w-3"
                                    style={{ backgroundColor: getColor(idx) }}
                                />
                                {categories?.[idx] ?? `Class ${idx + 1}`}
                            </span>
                        ))}
                    <span className="inline-flex items-center gap-1.5">
                        <span
                            aria-hidden
                            className="bg-foreground inline-block h-[2px] w-4"
                        />
                        train
                    </span>
                    {showTest && (
                        <span className="inline-flex items-center gap-1.5">
                            <svg
                                aria-hidden
                                width="16"
                                height="2"
                                className="text-foreground"
                            >
                                <line
                                    x1="0"
                                    y1="1"
                                    x2="16"
                                    y2="1"
                                    stroke="currentColor"
                                    strokeWidth="2"
                                    strokeDasharray="3 3"
                                />
                            </svg>
                            test
                        </span>
                    )}
                </span>
            </div>
        </div>
    );
}
