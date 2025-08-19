import MNISTGrid from '@/app/components/MNISTGrid';
import { useLogisticPlotData } from '@/app/hooks';
import type { DataState, TrainingReport } from '@/app/store';
import Plot from 'react-plotly.js';

type LogisticPlotsProps = {
    data: DataState;
    results: TrainingReport;
};

export default function LogisticPlots({ data, results }: LogisticPlotsProps) {
    const {
        trainTargetLabels,
        trainInputFeatures,
        testInputFeatures,
        testTargetLabels,
        categories,
        headers,
    } = data;

    const featureLength = trainInputFeatures[0]?.length || 0;

    const is2DPlot = featureLength === 2;
    const isMNISTPlot = featureLength >= 400; // MNIST has 28x28 or 20x20 pixels flattened to 784 features

    const { groupedData, groupedPredictions } = useLogisticPlotData(data);

    const { predictionPredictedLabels } = results;

    const predictions = [];

    if (groupedPredictions && predictionPredictedLabels) {
        const predictionsNum = 150;

        const classes = Array.from(new Set(predictionPredictedLabels.flat()));

        for (const cls of classes) {
            const z: number[][] = [];
            for (const index in predictionPredictedLabels) {
                const label = predictionPredictedLabels[index];
                const xIndex = Math.floor(Number(index) / predictionsNum);
                const yIndex = Number(index) % predictionsNum;
                if (!z[xIndex]) {
                    z[xIndex] = [];
                }

                z[xIndex][yIndex] = label[0] === cls ? 1 : 0;
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
            results.trainPredictedLabels && results.trainPredictedLabels.length
                ? [
                      ...results.trainPredictedLabels.slice(0, numbersToDisplay).flat(),
                      ...results.testPredictedLabels.slice(0, numbersToDisplay).flat(),
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
