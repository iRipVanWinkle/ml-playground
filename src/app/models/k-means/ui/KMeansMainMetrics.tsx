import type { KMeansTrainingReport } from '../types';
import type { MainMetricsProps } from '@/app/shared/registry';

export function KMeansMainMetrics({ report }: MainMetricsProps<KMeansTrainingReport>) {
    const { trainMetrics, testMetrics } = report;
    const trainSilhouetteScoreValue = trainMetrics?.silhouetteScore
        ? trainMetrics.silhouetteScore.toFixed(4)
        : '--';
    const testSilhouetteScoreValue = testMetrics?.silhouetteScore
        ? testMetrics.silhouetteScore.toFixed(4)
        : '--';

    return (
        <>
            <div>
                Train Silhouette: <div className="font-bold">{trainSilhouetteScoreValue}</div>
            </div>
            <div>
                Test Silhouette: <div className="font-bold">{testSilhouetteScoreValue}</div>
            </div>
        </>
    );
}
