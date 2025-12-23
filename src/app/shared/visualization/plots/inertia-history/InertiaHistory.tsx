import type { TrainingReport } from '@/app/models/types';
import { PlotlyScatter } from '../plotly';
import { useColor } from '../../colors';

type InertiaPlotProps = {
    report: TrainingReport;
};

export function InertiaHistory({ report }: InertiaPlotProps) {
    const { getColor } = useColor();
    const inertia = 'inertiaHistory' in report ? report.inertiaHistory : undefined;

    if (!inertia || inertia.length === 0) return null;

    return (
        <div className="w-full h-80">
            <PlotlyScatter
                data={[
                    {
                        x: inertia.map((_, i) => i + 1),
                        y: inertia,
                        mode: 'lines',
                        name: 'Inertia',
                        line: { color: getColor(0) },
                    },
                ]}
                layout={{
                    xaxis: { title: { text: 'Iterations' } },
                    yaxis: { title: { text: 'Inertia' } },
                    margin: { l: 40, r: 40, t: 20, b: 60 },
                }}
                style={{ width: '100%', height: '100%' }}
                useResizeHandler
            />
        </div>
    );
}
