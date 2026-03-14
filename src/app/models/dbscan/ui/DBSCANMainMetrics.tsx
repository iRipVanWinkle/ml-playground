import type { DBSCANTrainingReport } from '../types';
import type { MainMetricsProps } from '@/app/shared/registry';

export function DBSCANMainMetrics({ report }: MainMetricsProps<DBSCANTrainingReport>) {
    const { trainSilhouetteScore, testSilhouetteScore } = report;

    const trainSilhouetteScoreValue = trainSilhouetteScore ? trainSilhouetteScore.toFixed(4) : '--';
    const testSilhouetteScoreValue = testSilhouetteScore ? testSilhouetteScore.toFixed(4) : '--';

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
