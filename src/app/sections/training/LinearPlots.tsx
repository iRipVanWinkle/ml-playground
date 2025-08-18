import { useLinearPlotData } from '@/app/hooks';
import { type DataState, type TrainingReport } from '@/app/store';
import Plot from 'react-plotly.js';

type LinerPlotsProps = {
    data: DataState;
    report: TrainingReport;
};

export default function LinearPlots({ data, report }: LinerPlotsProps) {
    const { predictionPredictedLabels } = report;

    const { trainX, trainY, trainZ, testX, testY, testZ, predictionX, predictionY } =
        useLinearPlotData(data);

    const [yLabel, x1Label, x2Label] = data.headers;
    const is2DPlot = data.trainInputFeatures[0]?.length === 1;
    const is3DPlot = data.trainInputFeatures[0]?.length === 2;

    let plot = null;

    if (is2DPlot) {
        plot = (
            <Plot
                data={[
                    {
                        x: trainX,
                        y: trainY,
                        mode: 'markers',
                        name: 'Training Dataset',
                        marker: { color: 'green' },
                    },
                    {
                        x: testX,
                        y: testY,
                        mode: 'markers',
                        name: 'Test Dataset',
                        marker: { color: 'blue' },
                    },
                    ...(predictionPredictedLabels
                        ? [
                              {
                                  x: predictionX,
                                  y: predictionPredictedLabels?.flat(),
                                  mode: 'lines' as const,
                                  name: 'Prediction',
                                  line: { color: 'red' },
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
            <Plot
                data={[
                    {
                        x: trainX,
                        y: trainY,
                        z: trainZ,
                        mode: 'markers',
                        name: 'Training Dataset',
                        marker: { color: 'green' },
                        type: 'scatter3d',
                    },
                    {
                        x: testX,
                        y: testY,
                        z: testZ,
                        mode: 'markers',
                        name: 'Test Dataset',
                        marker: { color: 'blue' },
                        type: 'scatter3d',
                    },
                    ...(predictionPredictedLabels
                        ? [
                              {
                                  x: predictionX,
                                  y: predictionY,
                                  z: predictionPredictedLabels?.flat(),
                                  mode: 'lines' as const,
                                  name: 'Prediction',
                                  line: { color: 'red' },
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
