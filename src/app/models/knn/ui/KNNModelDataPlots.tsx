import { LinearPlots, LogisticPlots } from '@/app/shared/visualization';
import type { ModelDataPlotProps } from '@/app/shared/registry';
import type { KNNClassificationTrainingReport, KNNRegressionTrainingReport } from '../types';

export function KNNModelDataPlots({
    dataset,
    report,
}: ModelDataPlotProps<KNNClassificationTrainingReport | KNNRegressionTrainingReport>) {
    if (report.taskType === 'regression') {
        return <LinearPlots dataset={dataset} report={report} />;
    }

    return <LogisticPlots dataset={dataset} report={report} />;
}
