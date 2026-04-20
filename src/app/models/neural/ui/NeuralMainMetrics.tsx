import { AccuracyMetricsDisplay, LossMetricsDisplay } from '@/app/shared/visualization';
import type { NeuralClassificationTrainingReport, NeuralRegressionTrainingReport } from '../types';
import type { MainMetricsProps } from '@/app/shared/registry';

type NeuralTrainingReport = NeuralClassificationTrainingReport | NeuralRegressionTrainingReport;

export function NeuralMainMetrics({ report }: MainMetricsProps<NeuralTrainingReport>) {
    switch (report.taskType) {
        case 'regression':
            return <LossMetricsDisplay report={report} />;
        case 'classification':
            return <AccuracyMetricsDisplay report={report} />;
        default:
            return null;
    }
}
