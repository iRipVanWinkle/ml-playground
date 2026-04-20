import { MetricsPanel } from './MetricsPanel';
import { isNumber } from './utils';

interface R2MetricsDisplayProps {
    report: {
        trainMetrics?: { r2?: number | null } | null;
        testMetrics?: { r2?: number | null } | null;
    };
}

export function R2MetricsDisplay({ report }: R2MetricsDisplayProps) {
    const { trainMetrics, testMetrics } = report;

    const trainR2 = isNumber(trainMetrics?.r2) ? trainMetrics.r2.toFixed(4) : '--';
    const testR2 = isNumber(testMetrics?.r2) ? testMetrics.r2.toFixed(4) : '--';

    return (
        <MetricsPanel
            metrics={[
                { label: 'Train R²', value: trainR2 },
                { label: 'Test R²', value: testR2 },
            ]}
        />
    );
}
