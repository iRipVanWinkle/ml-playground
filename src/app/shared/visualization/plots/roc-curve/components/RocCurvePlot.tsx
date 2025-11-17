import Plot from 'react-plotly.js';
import type { RocCurveData } from '../types';
import { useRocCurvePlotData } from '../hooks';

type RocCurvePlotProps = {
    rocCurveData: RocCurveData;
    categories: string[];
};

export function RocCurvePlot({ rocCurveData, categories }: RocCurvePlotProps) {
    const plotData = useRocCurvePlotData({ rocCurveData, categories });

    return (
        <div className="w-full h-120 bg-background">
            <Plot
                data={plotData}
                layout={{
                    xaxis: {
                        title: { text: 'False Positive Rate' },
                        range: [-0.01, 1],
                        constrain: 'domain',
                        fixedrange: true,
                    },
                    yaxis: {
                        title: { text: 'True Positive Rate' },
                        range: [0, 1.02],
                        constrain: 'domain',
                        fixedrange: true,
                    },
                    legend: {
                        x: 0.5,
                        y: -0.2,
                        xanchor: 'center',
                        yanchor: 'top',
                        orientation: 'h',
                    },
                    margin: { l: 60, r: 40, t: 20, b: 20 },
                    hovermode: 'closest' as const,
                }}
                style={{ width: '100%', height: '100%' }}
                config={{ displayModeBar: false, staticPlot: false, responsive: true }}
                useResizeHandler
            />
        </div>
    );
}
