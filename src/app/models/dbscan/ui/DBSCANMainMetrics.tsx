import { AnomaliesMetricsDisplay, SilhouetteMetricsDisplay } from '@/app/shared/visualization';
import type { DBSCANTrainingReport } from '../types';
import type { MainMetricsProps } from '@/app/shared/registry';

export function DBSCANMainMetrics({ report }: MainMetricsProps<DBSCANTrainingReport>) {
    if (report.taskType === 'clustering') {
        return <SilhouetteMetricsDisplay report={report} />;
    }

    return <AnomaliesMetricsDisplay report={report} />;
}
