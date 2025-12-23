import type { TrainingReport } from '@/app/models/types';
import type { Dataset } from '@/app/shared/types';
import { useLinearPlotData } from './hooks/useLinearPlotData';
import { PlotlyScatter, PlotlyScatter3D } from '../plotly';
import { useColor } from '../../colors';

type LinerPlotsProps = {
    dataset: Dataset;
    report: TrainingReport;
};

export function LinearPlots({ dataset, report }: LinerPlotsProps) {
    const predictionPredictedLabels =
        'predictionPredictedLabels' in report ? report.predictionPredictedLabels : undefined;

    const { trainX, trainY, trainZ, testX, testY, testZ, predictionX, predictionY } =
        useLinearPlotData(dataset);

    const { getColor } = useColor();

    const trainColor = getColor(0);
    const testColor = getColor(1);
    const predictionColor = getColor(2);

    const [yLabel, x1Label, x2Label] = dataset.headers;
    const is2DPlot = dataset.trainInputFeatures[0]?.length === 1;
    const is3DPlot = dataset.trainInputFeatures[0]?.length === 2;

    let plot = null;

    if (is2DPlot) {
        plot = (
            <PlotlyScatter
                data={[
                    {
                        x: trainX,
                        y: trainY,
                        mode: 'markers',
                        name: 'Training Dataset',
                        marker: { color: trainColor },
                    },
                    {
                        x: testX,
                        y: testY,
                        mode: 'markers',
                        name: 'Test Dataset',
                        marker: { color: testColor },
                    },
                    ...(predictionPredictedLabels
                        ? [
                              {
                                  x: predictionX,
                                  y: predictionPredictedLabels.array,
                                  mode: 'lines' as const,
                                  name: 'Prediction',
                                  line: { color: predictionColor },
                              },
                          ]
                        : []),
                ]}
                layout={{
                    title: { text: 'Data & Model' },
                    xaxis: { title: { text: x1Label } },
                    yaxis: { title: { text: yLabel } },
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
                useResizeHandler
            />
        );
    } else if (is3DPlot) {
        plot = (
            <PlotlyScatter3D
                data={[
                    {
                        x: trainX,
                        y: trainY,
                        z: trainZ,
                        mode: 'markers',
                        name: 'Training Dataset',
                        marker: { color: trainColor },
                        type: 'scatter3d',
                    },
                    {
                        x: testX,
                        y: testY,
                        z: testZ,
                        mode: 'markers',
                        name: 'Test Dataset',
                        marker: { color: testColor },
                        type: 'scatter3d',
                    },
                    ...(predictionPredictedLabels
                        ? [
                              {
                                  x: predictionX,
                                  y: predictionY,
                                  z: predictionPredictedLabels.array,
                                  mode: 'lines' as const,
                                  name: 'Prediction',
                                  line: { color: predictionColor },
                                  type: 'scatter3d' as const,
                              },
                          ]
                        : []),
                ]}
                layout={{
                    title: { text: '3D Data & Model' },
                    scene: {
                        xaxis: { title: { text: x1Label } },
                        yaxis: { title: { text: x2Label } },
                        zaxis: { title: { text: yLabel } },
                    },
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
                useResizeHandler
            />
        );
    }

    return plot;
}
