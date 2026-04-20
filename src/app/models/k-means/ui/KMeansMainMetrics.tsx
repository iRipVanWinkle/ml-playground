import { SilhouetteMetricsDisplay } from '@/app/shared/visualization';
import type { KMeansTrainingReport } from '../types';
import type { MainMetricsProps } from '@/app/shared/registry';

export function KMeansMainMetrics({ report }: MainMetricsProps<KMeansTrainingReport>) {
    const { trainMetrics, testMetrics } = report;

    return (
        <SilhouetteMetricsDisplay
            report={{
                trainSilhouetteScore: trainMetrics?.silhouetteScore,
                testSilhouetteScore: testMetrics?.silhouetteScore,
            }}
        />
    );
}
