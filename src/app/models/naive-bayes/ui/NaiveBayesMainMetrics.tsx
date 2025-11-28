import type { NaiveBayesTrainingReport } from '../types';
import type { MainMetricsProps } from '@/app/shared/registry';

export function NaiveBayesMainMetrics({ report }: MainMetricsProps<NaiveBayesTrainingReport>) {
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
