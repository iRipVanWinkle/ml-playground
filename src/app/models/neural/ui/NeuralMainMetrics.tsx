import type { NeuralClassificationTrainingReport, NeuralRegressionTrainingReport } from '../types';
import type { MainMetricsProps } from '@/app/shared/registry';

type NeuralTrainingReport = NeuralClassificationTrainingReport | NeuralRegressionTrainingReport;

export function NeuralMainMetrics({ report }: MainMetricsProps<NeuralTrainingReport>) {
    switch (report.taskType) {
        case 'regression':
            return <NeuralRegressionMainMetrics report={report} />;
        case 'classification':
            return <NeuralClassificationMainMetrics report={report} />;
        default:
            return null;
    }
}

function NeuralRegressionMainMetrics({ report }: MainMetricsProps<NeuralRegressionTrainingReport>) {
    const { trainLoss, testLoss } = report;

    return (
        <>
            <div>
                Train Loss:{' '}
                <div className="font-bold">{trainLoss ? trainLoss.toFixed(4) : '--'}</div>
            </div>
            <div>
                Test Loss: <div className="font-bold">{testLoss ? testLoss.toFixed(4) : '--'}</div>
            </div>
        </>
    );
}

function NeuralClassificationMainMetrics({
    report,
}: MainMetricsProps<NeuralClassificationTrainingReport>) {
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
