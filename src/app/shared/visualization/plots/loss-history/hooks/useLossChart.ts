import { useMemo } from 'react';
import { line as d3line, scaleLinear } from 'd3';

type Size = { width: number; height: number };
type Padding = { top: number; right: number; bottom: number; left: number };

// A series source is either a single 1D series or many series stacked.
// Linear models return 1D (one output curve); multi-class models return 2D.
type SeriesInput = number[] | number[][] | undefined;

type Params = {
    train: SeriesInput;
    test: SeriesInput;
    size: Size;
    padding: Padding;
    yTickCount: number;
};

const EMPTY: number[][] = [];

function toSeriesMatrix(input: SeriesInput): number[][] {
    if (!input || input.length === 0) return EMPTY;
    return Array.isArray(input[0]) ? (input as number[][]) : [input as number[]];
}

export function useLossChart({
    train: rawTrain,
    test: rawTest,
    size,
    padding,
    yTickCount,
}: Params) {
    const train = useMemo(() => toSeriesMatrix(rawTrain), [rawTrain]);
    const test = useMemo(() => toSeriesMatrix(rawTest), [rawTest]);

    const stats = useMemo(() => {
        let yMaxRaw = 0;
        let maxLen = 0;

        for (let i = 0; i < train.length; i++) {
            const s = train[i];
            if (s.length > maxLen) maxLen = s.length;
            for (let j = 0; j < s.length; j++) {
                if (s[j] > yMaxRaw) yMaxRaw = s[j];
            }
        }
        for (let i = 0; i < test.length; i++) {
            const s = test[i];
            if (s.length > maxLen) maxLen = s.length;
            for (let j = 0; j < s.length; j++) {
                if (s[j] > yMaxRaw) yMaxRaw = s[j];
            }
        }

        return { yMaxRaw, maxLen };
    }, [train, test]);

    return useMemo(() => {
        if (!train.length) return null;

        const { yMaxRaw, maxLen } = stats;
        const yMax = Math.max(1, yMaxRaw * 1.1);
        const innerW = Math.max(1, size.width - padding.left - padding.right);
        const innerH = Math.max(1, size.height - padding.top - padding.bottom);

        const xScale = scaleLinear()
            .domain([0, Math.max(1, maxLen - 1)])
            .range([padding.left, padding.left + innerW]);
        const yScale = scaleLinear()
            .domain([0, yMax])
            .range([padding.top + innerH, padding.top]);

        const buildPath = d3line<number>()
            .x((_, i) => xScale(i))
            .y((v) => yScale(v));

        const yTicks = Array.from({ length: yTickCount + 1 }, (_, i) => ({
            value: yMax - (i / yTickCount) * yMax,
            y: padding.top + (i / yTickCount) * innerH,
        }));

        const trainPaths = train.map((vals) => buildPath(vals));
        const testPaths = test.map((vals) => buildPath(vals));
        const trainEnds = train.map((vals) =>
            vals.length
                ? { cx: xScale(vals.length - 1), cy: yScale(vals[vals.length - 1]) }
                : null,
        );

        return {
            train,
            test,
            maxLen,
            innerW,
            innerH,
            yTicks,
            trainPaths,
            testPaths,
            trainEnds,
        };
    }, [stats, train, test, size.width, size.height, padding, yTickCount]);
}
