import type { KNNClassificationTrainingReport, KNNRegressionTrainingReport } from '../types';
import type { MainMetricsProps } from '@/app/shared/registry';

type KNNTrainingReport = KNNClassificationTrainingReport | KNNRegressionTrainingReport;

export function KNNMainMetrics({ report }: MainMetricsProps<KNNTrainingReport>) {
    if (report.taskType === 'regression') {
        return <KNNRegressionMainMetrics report={report} />;
    }

    return <KNNClassificationMainMetrics report={report} />;
}

function KNNRegressionMainMetrics({ report }: MainMetricsProps<KNNRegressionTrainingReport>) {
    const { trainMetrics, testMetrics } = report;

    const trainR2 = trainMetrics?.r2 != null ? trainMetrics.r2.toFixed(4) : '--';
    const testR2 = testMetrics?.r2 != null ? testMetrics.r2.toFixed(4) : '--';

    return (
        <>
            <div>
                Train R²: <div className="font-bold tabular-nums">{trainR2}</div>
            </div>
            <div>
                Test R²: <div className="font-bold tabular-nums">{testR2}</div>
            </div>
        </>
    );
}

function KNNClassificationMainMetrics({
    report,
}: MainMetricsProps<KNNClassificationTrainingReport>) {
    const { trainAccuracy, testAccuracy } = report;

    const trainAccuracyValue =
        trainAccuracy != null ? (trainAccuracy * 100).toFixed(2) + '%' : '--';
    const testAccuracyValue = testAccuracy != null ? (testAccuracy * 100).toFixed(2) + '%' : '--';

    return (
        <>
            <div>
                Train Accuracy: <div className="font-bold tabular-nums">{trainAccuracyValue}</div>
            </div>
            <div>
                Test Accuracy: <div className="font-bold tabular-nums">{testAccuracyValue}</div>
            </div>
        </>
    );
}
