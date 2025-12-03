import type { TrainingReport } from '@/app/models/types';
import type { Dataset } from '@/app/shared/types';
import { PlotlyScatter } from '../plotly';

type LossHistoryProps = {
    dataset: Dataset;
    report: TrainingReport;
};

const colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'];

export function LossHistory({ dataset, report }: LossHistoryProps) {
    const lossHistory = 'trainLossHistory' in report ? report.trainLossHistory : undefined;
    const categories = dataset.categories;

    if (!lossHistory) return null;

    return (
        <div className="w-full h-80">
            <PlotlyScatter
                data={lossHistory.map((loss, index) => ({
                    x: Array.from({ length: loss.length }, (_, i) => i + 1),
                    y: loss,
                    mode: 'lines',
                    name: categories ? categories[index] : `Loss ${index + 1}`,
                    line: { color: colors[index % colors.length] },
                    marker: { color: colors[index % colors.length] },
                }))}
                layout={{
                    xaxis: { title: { text: 'Iterations' } },
                    yaxis: { title: { text: 'Loss' } },
                    legend: {
                        x: 0.5,
                        y: -0.4,
                        xanchor: 'center',
                        yanchor: 'top',
                        orientation: 'h',
                    },
                    margin: { l: 40, r: 40, t: 20, b: 60 },
                }}
                style={{ width: '100%', height: '100%' }}
                useResizeHandler
            />
        </div>
    );
}
