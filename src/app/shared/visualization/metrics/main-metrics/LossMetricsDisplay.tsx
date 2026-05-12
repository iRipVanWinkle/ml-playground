import { MetricsPanel } from './MetricsPanel';
import { isNumber } from './utils';

interface LossMetricsDisplayProps {
    report: {
        trainLoss?: number | null | undefined;
        testLoss?: number | null | undefined;
    };
}

export function LossMetricsDisplay({ report }: LossMetricsDisplayProps) {
    const { trainLoss, testLoss } = report;
    console.info(report);
    const trainLossValue = isNumber(trainLoss) ? trainLoss.toFixed(4) : '--';
    const testLossValue = isNumber(testLoss) ? testLoss.toFixed(4) : '--';

    return (
        <MetricsPanel
            metrics={[
                { label: 'Train Loss', value: trainLossValue },
                { label: 'Test Loss', value: testLossValue },
            ]}
        />
    );
}
