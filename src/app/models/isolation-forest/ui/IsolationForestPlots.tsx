import { useMemo } from 'react';
import type { Dataset } from '@/app/shared/types';
import type { IsolationForestTrainingReport } from '../types';
import { PlotlyScatter } from '@/app/shared/visualization/plots/plotly';
import { useColor } from '@/app/shared/visualization/colors';

type Props = {
    dataset: Dataset;
    report: IsolationForestTrainingReport;
};

export function IsolationForestPlots({ dataset, report }: Props) {
    const { trainInputFeatures, headers } = dataset;
    const { trainPredictions } = report;
    const { getColor } = useColor();

    const is2DPlot = trainInputFeatures[0]?.length === 2;
    const hasPredictions = (trainPredictions?.shape[0] ?? 0) > 0;

    const plotData = useMemo(() => {
        if (!is2DPlot) return [];

        const normal = { x: [] as number[], y: [] as number[] };
        const anomaly = { x: [] as number[], y: [] as number[] };

        if (hasPredictions) {
            for (let i = 0; i < trainInputFeatures.length; i++) {
                // predict returns -1 for anomaly, 1 for normal
                const isAnomaly = trainPredictions.array[i] === -1;
                const target = isAnomaly ? anomaly : normal;
                target.x.push(trainInputFeatures[i][0]);
                target.y.push(trainInputFeatures[i][1]);
            }
        } else {
            for (const p of trainInputFeatures) {
                normal.x.push(p[0]);
                normal.y.push(p[1]);
            }
        }

        return [
            {
                x: normal.x,
                y: normal.y,
                mode: 'markers' as const,
                name: 'Normal',
                marker: {
                    color: hasPredictions ? getColor(0) : '#9ca3af',
                    size: 6,
                    symbol: 'circle' as const,
                },
            },
            {
                x: anomaly.x,
                y: anomaly.y,
                mode: 'markers' as const,
                name: 'Anomaly',
                marker: { color: getColor(1), size: 8, symbol: 'x' as const },
            },
        ];
    }, [trainInputFeatures, trainPredictions, hasPredictions, is2DPlot, getColor]);

    if (!is2DPlot) {
        return (
            <p className="text-sm text-muted-foreground p-4">
                Plotting requires exactly 2 input features
            </p>
        );
    }

    return (
        <PlotlyScatter
            data={plotData}
            layout={{
                xaxis: { title: { text: headers[1] } },
                yaxis: { title: { text: headers[2] } },
                showlegend: true,
            }}
        />
    );
}
