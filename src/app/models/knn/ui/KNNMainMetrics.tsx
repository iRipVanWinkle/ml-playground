import { AccuracyMetricsDisplay, R2MetricsDisplay } from '@/app/shared/visualization';
import type { KNNClassificationTrainingReport, KNNRegressionTrainingReport } from '../types';
import type { MainMetricsProps } from '@/app/shared/registry';

type KNNTrainingReport = KNNClassificationTrainingReport | KNNRegressionTrainingReport;

export function KNNMainMetrics({ report }: MainMetricsProps<KNNTrainingReport>) {
    if (report.taskType === 'regression') {
        return <R2MetricsDisplay report={report} />;
    }

    return <AccuracyMetricsDisplay report={report} />;
}
