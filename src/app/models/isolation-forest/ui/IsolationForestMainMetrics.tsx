import type { IsolationForestTrainingReport } from '../types';
import type { MainMetricsProps } from '@/app/shared/registry';

export function IsolationForestMainMetrics({
    report,
}: MainMetricsProps<IsolationForestTrainingReport>) {
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
