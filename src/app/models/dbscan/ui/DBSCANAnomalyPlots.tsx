import type { Dataset } from '@/app/shared/types';
import type { DBSCANSettings, DBSCANAnomalyTrainingReport } from '../types';
import { PlotlyScatter } from '@/app/shared/visualization/plots/plotly';
import { useColor } from '@/app/shared/visualization/colors';
import { useDBSCANPlotData } from './hooks/useDBSCANPlotData';
import { useEpsilonShapes } from './hooks/useEpsilonShapes';
import { useAxisRanges } from './hooks/useAxisRanges';

type DBSCANPlotsProps = {
    dataset: Dataset;
    report: DBSCANAnomalyTrainingReport;
    modelSettings: DBSCANSettings;
};

export function DBSCANAnomalyPlots({ dataset, report, modelSettings }: DBSCANPlotsProps) {
    const { trainInputFeatures, testInputFeatures, headers } = dataset;
    const { epsilon, distance } = modelSettings;
    const { trainPredictions, testPredictions, numClusters, activePointIndex } = report;
    const { getColor } = useColor();

    const hasAssignments = report.type === 'dbscan' && (trainPredictions?.shape[0] ?? 0) > 0;
    const is2DPlot = trainInputFeatures[0]?.length === 2;

    const [, x1Label, x2Label] = headers;

    const data2D = useDBSCANPlotData({
        trainInputFeatures,
        testInputFeatures,
        trainPredictions,
        testPredictions,
        numClusters,
        activePointIndex,
        hasAssignments,
        getColor,
    });
    const epsilonCircleShapes = useEpsilonShapes({
        epsilon,
        distance,
        activePointIndex,
        trainInputFeatures,
    });
    const axisRanges2D = useAxisRanges({ trainInputFeatures, testInputFeatures, epsilon });

    if (is2DPlot) {
        return (
            <PlotlyScatter
                data={data2D}
                layout={{
                    title: { text: 'Data & Model' },
                    xaxis: {
                        title: { text: x1Label },
                        ...(axisRanges2D && { range: axisRanges2D.x, autorange: false }),
                    },
                    yaxis: {
                        title: { text: x2Label },
                        ...(axisRanges2D && { range: axisRanges2D.y, autorange: false }),
                    },
                    shapes: epsilonCircleShapes,
                }}
            />
        );
    }

    return <p className="text-sm text-muted-foreground p-4">Plotting requires 2 input features</p>;
}
