import type { TreeClassificationTrainingReport, TreeRegressionTrainingReport } from '../types';
import type { MainMetricsProps } from '@/app/shared/registry';

type TreeTrainingReport = TreeClassificationTrainingReport | TreeRegressionTrainingReport;

export function TreeMainMetrics({ report }: MainMetricsProps<TreeTrainingReport>) {
    switch (report.taskType) {
        case 'regression':
            return <TreeRegressionMainMetrics report={report} />;
        case 'classification':
            return <TreeClassificationMainMetrics report={report} />;
        default:
            return null;
    }
}

function TreeRegressionMainMetrics({ report }: MainMetricsProps<TreeRegressionTrainingReport>) {
    const { testLoss } = report;

    return (
        <>
            <div>
                Train Loss: <div className="font-bold">--</div>
            </div>
            <div>
                Test Loss: <div className="font-bold">{testLoss ? testLoss.toFixed(4) : '--'}</div>
            </div>
        </>
    );
}

function TreeClassificationMainMetrics({
    report,
}: MainMetricsProps<TreeClassificationTrainingReport>) {
    const { trainAccuracy, testAccuracy } = report;

    const trainAccuracyValue = trainAccuracy ? (trainAccuracy * 100).toFixed(2) + '%' : '--';
    const testAccuracyValue = testAccuracy ? (testAccuracy * 100).toFixed(2) + '%' : '--';

    return (
        <>
            <div>
                Train Accuracy: <div className="font-bold">{trainAccuracyValue}</div>
            </div>
            <div>
                Test Accuracy: <div className="font-bold">{testAccuracyValue}</div>
            </div>
        </>
    );
}
