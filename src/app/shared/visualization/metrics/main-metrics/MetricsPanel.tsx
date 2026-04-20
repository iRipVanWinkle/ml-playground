export interface MetricItem {
    label: string;
    value: string;
    testId?: string;
}

interface MetricsPanelProps {
    metrics: MetricItem[];
}

export function MetricsPanel({ metrics }: MetricsPanelProps) {
    return metrics.map(({ label, value, testId }) => (
        <div key={label}>
            {label}:{' '}
            <div className="font-bold tabular-nums" data-testid={testId}>
                {value}
            </div>
        </div>
    ));
}
