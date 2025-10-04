import Plot from 'react-plotly.js';

type LossHistoryPlotProps = {
    lossHistory: number[][];
    categories?: string[];
};

const colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'];

export function LossHistoryPlot({ lossHistory, categories }: LossHistoryPlotProps) {
    return (
        <Plot
            data={lossHistory.map((loss, index) => ({
                x: Array.from({ length: loss.length }, (_, i) => i + 1),
                y: loss,
                mode: 'lines',
                name: categories ? categories[index] : `Loss ${index + 1}`,
                line: { color: colors[index % colors.length] },
                marker: { color: colors[index % colors.length] },
            }))}
            layout={{
                title: { text: 'Loss History' },
                xaxis: { title: { text: 'Iterations' } },
                yaxis: { title: { text: 'Loss' } },
                legend: { x: 0.5, y: -0.4, xanchor: 'center', yanchor: 'top', orientation: 'h' },
                margin: { l: 40, r: 40, t: 60, b: 60 },
            }}
            style={{ width: '100%', height: '100%' }}
            useResizeHandler
        />
    );
}
