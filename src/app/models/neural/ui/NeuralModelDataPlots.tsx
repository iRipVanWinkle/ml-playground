import { LinearPlots, LogisticPlots } from '@/app/shared/visualization';
import type { ModelDataPlotProps } from '@/app/shared/registry';
import type { NeuralClassificationTrainingReport, NeuralRegressionTrainingReport } from '../types';

export function NeuralModelDataPlots({
    dataset,
    report,
}: ModelDataPlotProps<NeuralClassificationTrainingReport | NeuralRegressionTrainingReport>) {
    if (report.taskType === 'regression') {
        return <LinearPlots dataset={dataset} report={report} />;
    }

    if (report.taskType === 'classification') {
        return <LogisticPlots dataset={dataset} report={report} />;
    }

    return null;
}
