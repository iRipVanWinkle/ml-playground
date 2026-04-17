import type { DBSCANTrainingReport } from '../types';
import type { MainMetricsProps } from '@/app/shared/registry';

export function DBSCANMainMetrics({ report }: MainMetricsProps<DBSCANTrainingReport>) {
    if (report.taskType === 'clustering') {
        const { trainSilhouetteScore, testSilhouetteScore } = report;
        const trainSilhouetteScoreValue = trainSilhouetteScore
            ? trainSilhouetteScore.toFixed(4)
            : '--';
        const testSilhouetteScoreValue = testSilhouetteScore
            ? testSilhouetteScore.toFixed(4)
            : '--';

        return (
            <>
                <div>
                    Train Silhouette:{' '}
                    <div className="font-bold tabular-nums">{trainSilhouetteScoreValue}</div>
                </div>
                <div>
                    Test Silhouette:{' '}
                    <div className="font-bold tabular-nums">{testSilhouetteScoreValue}</div>
                </div>
            </>
        );
    }

    const { trainAnomalyRate, testAnomalyRate } = report;
    const trainAnomalyRateValue =
        trainAnomalyRate !== undefined ? (trainAnomalyRate * 100).toFixed(2) + '%' : '--';
    const testAnomalyRateValue =
        testAnomalyRate !== undefined ? (testAnomalyRate * 100).toFixed(2) + '%' : '--';

    return (
        <>
            <div>
                Train Anomalies:{' '}
                <div className="font-bold tabular-nums">{trainAnomalyRateValue}</div>
            </div>
            <div>
                Test Anomalies: <div className="font-bold tabular-nums">{testAnomalyRateValue}</div>
            </div>
        </>
    );
}
