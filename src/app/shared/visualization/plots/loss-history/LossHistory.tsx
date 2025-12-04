import type { TrainingReport } from '@/app/models/types';
import type { Dataset } from '@/app/shared/types';
import { PlotlyScatter } from '../plotly';
import { useColor } from '../../colors';

type LossHistoryProps = {
    dataset: Dataset;
    report: TrainingReport;
};

export function LossHistory({ dataset, report }: LossHistoryProps) {
    const { getColor } = useColor();
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
                    line: { color: getColor(index) },
                    marker: { color: getColor(index) },
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
            />
        </div>
    );
}
