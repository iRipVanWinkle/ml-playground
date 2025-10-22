import { LinearPlots, LogisticPlots } from '@/app/shared/visualization';
import type { ModelDataPlotProps } from '@/app/shared/registry';
import type { TreeClassificationTrainingReport, TreeRegressionTrainingReport } from '../types';

export function TreeModelDataPlots({
    dataset,
    report,
}: ModelDataPlotProps<TreeClassificationTrainingReport | TreeRegressionTrainingReport>) {
    if (report.taskType === 'regression') {
        return <LinearPlots dataset={dataset} report={report} />;
    }

    if (report.taskType === 'classification') {
        return <LogisticPlots dataset={dataset} report={report} />;
    }

    return null;
}
