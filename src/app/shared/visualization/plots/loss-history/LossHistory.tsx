import type { TrainingReport } from '@/app/models/types';
import type { Dataset } from '@/app/shared/types';
import { useColor } from '../../colors';
import { useContainerSize } from './hooks/useContainerSize';
import { useLossChart } from './hooks/useLossChart';

type LossHistoryProps = {
    dataset: Dataset;
    report: TrainingReport;
};

const PAD = { top: 16, right: 20, bottom: 30, left: 50 };
const Y_TICKS = 4;
const FALLBACK_SIZE = { width: 820, height: 260 };

export function LossHistory({ dataset, report }: LossHistoryProps) {
    const { getColor } = useColor();
    const { containerRef, size } = useContainerSize(FALLBACK_SIZE);

    const chart = useLossChart({
        train: 'trainLossHistory' in report ? report.trainLossHistory : undefined,
        test: 'testLossHistory' in report ? report.testLossHistory : undefined,
        size,
        padding: PAD,
        yTickCount: Y_TICKS,
    });

    if (!chart) return null;

    const { train, test, maxLen, yTicks, innerW, innerH, trainPaths, testPaths, trainEnds } =
        chart;
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
                        testPaths.map((path, idx) =>
                            path ? (
                                <path
                                    key={`test-${idx}`}
                                    d={path}
                                    fill="none"
                                    stroke={getColor(idx)}
                                    strokeWidth={1.5}
                                    strokeDasharray="3 3"
                                    opacity={0.85}
                                />
                            ) : null,
                        )}

                    {trainPaths.map((path, idx) =>
                        path ? (
                            <path
                                key={`train-${idx}`}
                                d={path}
                                fill="none"
                                stroke={getColor(idx)}
                                strokeWidth={1.75}
                            />
                        ) : null,
                    )}

                    {trainEnds.map((end, idx) =>
                        end ? (
                            <circle
                                key={`dot-${idx}`}
                                cx={end.cx}
                                cy={end.cy}
                                r={3}
                                className="fill-background"
                                stroke={getColor(idx)}
                                strokeWidth={1.5}
                            />
                        ) : null,
                    )}
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
