import { useMNISTGridFrame } from '@/app/hooks';
import React from 'react';
import Plot from 'react-plotly.js';

interface MNISTGridProps {
    data: number[][];
    predictions?: number[];
    labels: number[];
    originalLabels: string[];
}

const MNISTGrid: React.FC<MNISTGridProps> = ({ data, labels, originalLabels, predictions }) => {
    const numbersToDisplay = data.length;
    const frames = useMNISTGridFrame(data);

    const ind = (index: number) => (index ? index + 1 : '');

    const plotData: Partial<Plotly.PlotData>[] = labels.map((digitLabel, index) => {
        const predictedLabel = predictions?.[index];

        const colorscale: [number, string][] = predictions
            ? predictedLabel === digitLabel
                ? [
                      [0, 'white'],
                      [1, 'green'],
                  ]
                : [
                      [0, 'white'],
                      [1, 'red'],
                  ]
            : [
                  [0, 'white'],
                  [1, 'black'],
              ];

        return {
            z: frames[index],
            type: 'heatmap',
            colorscale,
            zsmooth: false,
            showscale: false,
            xaxis: `x${ind(index)}`,
            yaxis: `y${ind(index)}`,
            hoverinfo: 'none',
        } as Partial<Plotly.PlotData>;
    });

    const layout: Partial<Plotly.Layout> = {
        showlegend: false,
        legend: { x: 0.5, y: -0.2, xanchor: 'center', yanchor: 'top', orientation: 'h' },
        margin: { l: 10, r: 20, t: 10, b: 20 },
        grid: { rows: 8, columns: 5, pattern: 'independent', xgap: 0.25, ygap: 0.25 },
        annotations: labels.map((label, index) => ({
            text:
                originalLabels[label] +
                (predictions ? ` (${originalLabels[predictions![index]]})` : ''),
            xref: `x${ind(index)} domain`,
            yref: `y${ind(index)} domain`,
            x: 0.5,
            y: -0.3,
            showarrow: false,
            font: { size: 12, color: 'black', align: 'center' },
        })),
        ...Object.fromEntries(
            Array.from({ length: numbersToDisplay }, (_, index) => {
                return [
                    [
                        `xaxis${ind(index)}`,
                        {
                            showgrid: false,
                            visible: false,
                            fixedrange: true,
                            scaleanchor: `y${ind(index)}`,
                        },
                    ],
                    [
                        `yaxis${ind(index)}`,
                        {
                            showgrid: false,
                            visible: false,
                            fixedrange: true,
                            anchor: `x${ind(index)}`,
                        },
                    ],
                ];
            }).flat(),
        ),
    };

    return (
        <Plot
            data={plotData}
            layout={layout}
            config={{ displayModeBar: false, staticPlot: true, responsive: true }}
            style={{ width: '100%', aspectRatio: '1 / 1' }}
        />
    );
};

export default MNISTGrid;
