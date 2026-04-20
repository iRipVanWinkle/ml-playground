import { MetricsPanel } from './MetricsPanel';
import { isNumber } from './utils';

interface SilhouetteMetricsDisplayProps {
    report: {
        trainSilhouetteScore?: number | null | undefined;
        testSilhouetteScore?: number | null | undefined;
    };
}

export function SilhouetteMetricsDisplay({ report }: SilhouetteMetricsDisplayProps) {
    const { trainSilhouetteScore, testSilhouetteScore } = report;

    const trainSilhouetteScoreValue = isNumber(trainSilhouetteScore)
        ? trainSilhouetteScore.toFixed(4)
        : '--';
    const testSilhouetteScoreValue = isNumber(testSilhouetteScore)
        ? testSilhouetteScore.toFixed(4)
        : '--';

    return (
        <MetricsPanel
            metrics={[
                { label: 'Train Silhouette', value: trainSilhouetteScoreValue },
                { label: 'Test Silhouette', value: testSilhouetteScoreValue },
            ]}
        />
    );
}
