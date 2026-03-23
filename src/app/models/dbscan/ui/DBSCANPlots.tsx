import type { Dataset } from '@/app/shared/types';
import type { DBSCANSettings, DBSCANTrainingReport } from '../types';
import { DBSCANClusteringPlots } from './DBSCANClusteringPlots';
import { DBSCANAnomalyPlots } from './DBSCANAnomalyPlots';

type DBSCANPlotsProps = {
    dataset: Dataset;
    report: DBSCANTrainingReport;
    modelSettings: DBSCANSettings;
};

export function DBSCANPlots({ dataset, report, modelSettings }: DBSCANPlotsProps) {
    if (report.taskType === 'clustering') {
        return (
            <DBSCANClusteringPlots
                dataset={dataset}
                report={report}
                modelSettings={modelSettings}
            />
        );
    } else {
        return (
            <DBSCANAnomalyPlots dataset={dataset} report={report} modelSettings={modelSettings} />
        );
    }
}
