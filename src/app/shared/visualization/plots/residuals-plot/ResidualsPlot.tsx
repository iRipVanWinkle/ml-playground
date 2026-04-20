import { useState } from 'react';
import type { TrainingReport } from '@/app/models/types';
import { TrainTestSelector } from '@/app/shared/ui';
import { PlotlyScatter } from '../plotly';
import { useColor } from '../../colors';

type ResidualsPlotProps = {
    report: TrainingReport;
};

export function ResidualsPlot({ report }: ResidualsPlotProps) {
    const [selectedDataset, setSelectedDataset] = useState<string>('train');
    const { getColor } = useColor();

    const supportsResiduals = 'trainResiduals' in report && report.taskType === 'regression';

    if (!supportsResiduals) {
        return (
            <div className="w-full h-full p-4 flex items-center justify-center">
                <div className="text-center text-muted-foreground">
                    This model does not support residuals plot.
                </div>
            </div>
        );
    }

    const hasResiduals = report.trainResiduals.array.length > 0;

    if (!hasResiduals) {
        return (
            <div className="w-full h-full p-4 flex items-center justify-center">
                <div className="text-sm text-center text-muted-foreground">
                    Run training to see residuals plot
                </div>
            </div>
        );
    }

    const residuals =
        selectedDataset === 'test' && report.testResiduals
            ? report.testResiduals
            : report.trainResiduals;

    const predictions =
        selectedDataset === 'test' && report.testPredictedLabels
            ? report.testPredictedLabels.array
            : report.trainPredictedLabels.array;

    const residualsFlat = residuals.array;

    return (
        <div className="w-full py-4 bg-card">
            {report.testResiduals && (
                <div className="flex flex-row justify-end mb-4 px-4">
                    <TrainTestSelector value={selectedDataset} onChange={setSelectedDataset} />
                </div>
            )}

            <div className="w-full h-80 px-4">
                <PlotlyScatter
                    data={[
                        {
                            x: predictions,
                            y: residualsFlat,
                            mode: 'markers',
                            type: 'scatter',
                            marker: {
                                color: getColor(0),
                                size: 6,
                            },
                            name: 'Residuals',
                        },
                        ...(predictions.length > 0
                            ? [
                                  {
                                      x: [Math.min(...predictions), Math.max(...predictions)],
                                      y: [0, 0],
                                      mode: 'lines' as const,
                                      type: 'scatter' as const,
                                      line: {
                                          color: getColor(3), // Red
                                          width: 2,
                                          dash: 'dash' as const,
                                      },
                                      name: 'Zero Line',
                                      showlegend: false,
                                  },
                              ]
                            : []),
                    ]}
                    layout={{
                        xaxis: { title: { text: 'Predicted Values' } },
                        yaxis: { title: { text: 'Residuals' } },
                        hovermode: 'closest',
                        legend: {
                            x: 0.5,
                            y: -0.3,
                            xanchor: 'center',
                            yanchor: 'top',
                            orientation: 'h',
                        },
                        margin: { l: 60, r: 40, t: 20, b: 20 },
                    }}
                />
            </div>
        </div>
    );
}
