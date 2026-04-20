import { AccuracyMetricsDisplay, R2MetricsDisplay } from '@/app/shared/visualization';
import type { TreeClassificationTrainingReport, TreeRegressionTrainingReport } from '../types';
import type { MainMetricsProps } from '@/app/shared/registry';

type TreeTrainingReport = TreeClassificationTrainingReport | TreeRegressionTrainingReport;

export function TreeMainMetrics({ report }: MainMetricsProps<TreeTrainingReport>) {
    switch (report.taskType) {
        case 'regression':
            return <R2MetricsDisplay report={report} />;
        case 'classification':
            return <AccuracyMetricsDisplay report={report} />;
        default:
            return null;
    }
}
