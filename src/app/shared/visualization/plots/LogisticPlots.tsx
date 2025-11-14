import MNISTGrid from './MNISTGrid';
import type { TrainingReport } from '@/app/models/types';
import type { Dataset } from '@/app/shared/types';
import Plot from 'react-plotly.js';
import { useLogisticPlotData } from './hooks/useLogisticPlotData';

type LogisticPlotsProps = {
    dataset: Dataset;
    report: TrainingReport;
};

export function LogisticPlots({ dataset, report }: LogisticPlotsProps) {
    const {
        trainTargetLabels,
        trainInputFeatures,
        testInputFeatures,
        testTargetLabels,
        categories,
        headers,
    } = dataset;

    const featureLength = trainInputFeatures[0]?.length || 0;

    const is2DPlot = featureLength === 2;
    const isMNISTPlot = dataset.isImage;
    const { groupedData, groupedPredictions } = useLogisticPlotData(dataset);

    const { predictionPredictedLabels } = report ?? {};

    const predictions = [];

    if (groupedPredictions && predictionPredictedLabels) {
        const predictionsNum = 150;

        const classes = Array.from(new Set(predictionPredictedLabels.array));

        for (const cls of classes) {
            const z: number[][] = [];
            for (const index in predictionPredictedLabels.array) {
                const label = predictionPredictedLabels.array[index];
                const xIndex = Math.floor(Number(index) / predictionsNum);
                const yIndex = Number(index) % predictionsNum;
                if (!z[xIndex]) {
                    z[xIndex] = [];
                }

                z[xIndex][yIndex] = label === cls ? 1 : 0;
            }
            predictions.push({
                x: groupedPredictions.predictionX,
                y: groupedPredictions.predictionY,
                z,
            });
        }
    }

    const colors = ['green', 'blue', 'red', 'orange', 'purple', 'cyan'];
    const treningColors = ['lime', 'skyblue', 'crimson', 'gold', 'violet', 'teal'];

    let plot = null;

    if (is2DPlot) {
        const plotData = [
            ...predictions.map((pred) => ({
                x: pred.x,
                y: pred.y,
                z: pred.z,
                type: 'contour',
                showscale: false,
                showlegend: false,
                contours: {
                    coloring: 'lines',
                    showscale: false,
                },
            })),
            ...Object.entries(groupedData)
                .map(([label, points]) => [
                    {
                        x: points.trainingX,
                        y: points.trainingY,
                        mode: 'markers',
                        name: `${categories![Number(label)]}`,
                        marker: { color: colors[Number(label)] },
                        legendgroup: 'Training Dataset',
                    },
                    {
                        x: points.testingX,
                        y: points.testingY,
                        mode: 'markers',
                        name: `${categories![Number(label)]}`,
                        marker: { color: treningColors[Number(label)] },
                        legendgroup: 'Test Dataset',
                    },
                ])
                .flat(),
        ] as Partial<Plotly.PlotData>[];

        plot = (
            <Plot
                data={plotData}
                layout={{
                    title: { text: 'Data & Model' },
                    xaxis: { title: { text: headers[1] } },
                    yaxis: { title: { text: headers[2] } },
                    legend: {
                        x: 0.5,
                        y: -0.2,
                        xanchor: 'center',
                        yanchor: 'top',
                        orientation: 'h',
                    },
                    margin: { l: 40, r: 40, t: 40, b: 40 },
                }}
                style={{ width: '100%', height: '100%' }}
            />
        );
    }

    if (isMNISTPlot) {
        const numbersToDisplay = 20; // Display 32 digits in the grid
        const dataPlot = [
            ...trainInputFeatures.slice(0, numbersToDisplay),
            ...testInputFeatures.slice(0, numbersToDisplay),
        ];

        const predictionsPlot =
            report.trainPredictedLabels && report.trainPredictedLabels.array.length
                ? [
                      ...report.trainPredictedLabels.array.slice(0, numbersToDisplay),
                      ...report.testPredictedLabels.array.slice(0, numbersToDisplay),
                  ]
                : undefined;

        const labelsPlot = [
            ...trainTargetLabels.slice(0, numbersToDisplay).flat(),
            ...testTargetLabels.slice(0, numbersToDisplay).flat(),
        ];

        plot = (
            <MNISTGrid
                data={dataPlot}
                predictions={predictionsPlot}
                labels={labelsPlot}
                originalLabels={categories!}
            />
        );
    }

    return plot;
}
