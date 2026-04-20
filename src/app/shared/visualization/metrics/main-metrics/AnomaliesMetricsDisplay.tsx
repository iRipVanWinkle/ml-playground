import { MetricsPanel } from './MetricsPanel';
import { isNumber } from './utils';

interface AnomaliesMetricsDisplayProps {
    report: {
        trainAnomalyRate?: number | null | undefined;
        testAnomalyRate?: number | null | undefined;
    };
}

export function AnomaliesMetricsDisplay({ report }: AnomaliesMetricsDisplayProps) {
    const { trainAnomalyRate, testAnomalyRate } = report;

    const trainAnomalyRateValue = isNumber(trainAnomalyRate)
        ? (trainAnomalyRate * 100).toFixed(2) + '%'
        : '--';
    const testAnomalyRateValue = isNumber(testAnomalyRate)
        ? (testAnomalyRate * 100).toFixed(2) + '%'
        : '--';

    return (
        <MetricsPanel
            metrics={[
                { label: 'Train Anomalies', value: trainAnomalyRateValue },
                { label: 'Test Anomalies', value: testAnomalyRateValue },
            ]}
        />
    );
}
