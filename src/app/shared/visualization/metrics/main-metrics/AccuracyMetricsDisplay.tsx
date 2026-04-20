import { MetricsPanel } from './MetricsPanel';
import { isNumber } from './utils';

interface AccuracyMetricsDisplayProps {
    report: {
        trainAccuracy?: number | null | undefined;
        testAccuracy?: number | null | undefined;
    };
}

export function AccuracyMetricsDisplay({ report }: AccuracyMetricsDisplayProps) {
    const { trainAccuracy, testAccuracy } = report;

    const trainAccuracyValue = isNumber(trainAccuracy)
        ? (trainAccuracy * 100).toFixed(2) + '%'
        : '--';
    const testAccuracyValue = isNumber(testAccuracy) ? (testAccuracy * 100).toFixed(2) + '%' : '--';

    return (
        <MetricsPanel
            metrics={[
                { label: 'Train Accuracy', value: trainAccuracyValue },
                { label: 'Test Accuracy', value: testAccuracyValue },
            ]}
        />
    );
}
