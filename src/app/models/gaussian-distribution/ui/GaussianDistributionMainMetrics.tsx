import type { GaussianDistributionTrainingReport } from '../types';
import type { MainMetricsProps } from '@/app/shared/registry';

export function GaussianDistributionMainMetrics({
    report,
}: MainMetricsProps<GaussianDistributionTrainingReport>) {
    const { trainAnomalyRate, testAnomalyRate } = report;

    const trainValue = (trainAnomalyRate * 100).toFixed(2) + '%';
    const testValue =
        testAnomalyRate !== undefined ? (testAnomalyRate * 100).toFixed(2) + '%' : '--';

    return (
        <>
            <div>
                Train Anomalies: <div className="font-bold tabular-nums">{trainValue}</div>
            </div>
            <div>
                Test Anomalies: <div className="font-bold tabular-nums">{testValue}</div>
            </div>
        </>
    );
}
