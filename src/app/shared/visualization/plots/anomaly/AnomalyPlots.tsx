import type { BaseAnomalyReport, Dataset } from '@/app/shared/types';
import { PlotlyScatter } from '../plotly';
import { useColor } from '../../colors';
import { useAnomalyPartition } from './useAnomalyPartition';

export type AnomalyPlotsProps = {
    dataset: Dataset;
    report: BaseAnomalyReport;
};

export function AnomalyPlots({ dataset, report }: AnomalyPlotsProps) {
    const { trainInputFeatures, testInputFeatures, headers } = dataset;
    const { trainPredictions, testPredictions } = report;
    const { getColor } = useColor();

    const {
        normal: trainNormal,
        anomaly: trainAnomaly,
        hasPredictions: hasTrainPredictions,
    } = useAnomalyPartition(trainInputFeatures, trainPredictions);
    const {
        normal: testNormal,
        anomaly: testAnomaly,
        hasPredictions: hasTestPredictions,
    } = useAnomalyPartition(testInputFeatures, testPredictions);

    const is2DPlot = trainInputFeatures[0]?.length === 2;

    const plotData = [
        {
            x: trainNormal.x,
            y: trainNormal.y,
            mode: 'markers',
            name: 'Train Normal',
            marker: {
                color: hasTrainPredictions ? getColor(0) : 'grey',
                symbol: 'circle',
            },
        },
        {
            x: trainAnomaly.x,
            y: trainAnomaly.y,
            mode: 'markers',
            name: 'Train Anomaly',
            marker: { color: getColor('red'), symbol: 'x' },
        },
        {
            x: testNormal.x,
            y: testNormal.y,
            mode: 'markers',
            name: 'Test Normal',
            marker: {
                color: hasTestPredictions ? getColor(0) : 'grey',
                symbol: 'circle-open',
            },
        },
        {
            x: testAnomaly.x,
            y: testAnomaly.y,
            mode: 'markers',
            name: 'Test Anomaly',
            marker: { color: getColor('red'), symbol: 'x-open' },
        },
    ];

    let plot = (
        <p className="text-sm text-muted-foreground p-4">Plotting requires 2 input features</p>
    );

    if (is2DPlot) {
        plot = (
            <PlotlyScatter
                data={plotData}
                layout={{
                    title: { text: 'Data & Model' },
                    xaxis: { title: { text: headers[1] } },
                    yaxis: { title: { text: headers[2] } },
                }}
            />
        );
    }

    return plot;
}
