import { useMemo } from 'react';
import type { DBSCANSettings } from '../../types';

type UseEpsilonShapesProps = {
    trainInputFeatures: number[][];
    distance: DBSCANSettings['distance'];
    epsilon: number;
    activePointIndex?: number;
};

export function useEpsilonShapes({
    epsilon,
    distance,
    activePointIndex,
    trainInputFeatures,
}: UseEpsilonShapesProps) {
    return useMemo(() => {
        const is2DPlot = trainInputFeatures[0].length === 2;
        if (
            !is2DPlot ||
            activePointIndex === undefined ||
            distance.type === 'cosine' ||
            activePointIndex >= trainInputFeatures.length
        ) {
            return [];
        }

        const cx = trainInputFeatures[activePointIndex][0];
        const cy = trainInputFeatures[activePointIndex][1];

        const sharedStyle = {
            xref: 'x' as const,
            yref: 'y' as const,
            line: { color: 'rgba(239, 68, 68, 0.4)', dash: 'dash' as const, width: 1.5 },
            fillcolor: 'rgba(239, 68, 68, 0.06)',
        };

        if (distance.type === 'euclidean') {
            return [
                {
                    type: 'circle' as const,
                    x0: cx - epsilon,
                    y0: cy - epsilon,
                    x1: cx + epsilon,
                    y1: cy + epsilon,
                    ...sharedStyle,
                },
            ];
        }

        // Manhattan: L1 ball is a diamond in data coordinates
        return [
            {
                type: 'path' as const,
                path: `M ${cx},${cy - epsilon} L ${cx + epsilon},${cy} L ${cx},${cy + epsilon} L ${cx - epsilon},${cy} Z`,
                ...sharedStyle,
            },
        ];
    }, [activePointIndex, epsilon, distance, trainInputFeatures]);
}
