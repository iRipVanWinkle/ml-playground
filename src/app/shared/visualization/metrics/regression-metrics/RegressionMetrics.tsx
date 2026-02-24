import { useEffect, useState } from 'react';
import type { TrainingReport } from '@/app/models/types';
import { TrainTestSelector } from '@/app/shared/ui';
import { MetricsDisplay } from './components';
import type { Dataset } from '@/app/shared/types';

interface RegressionMetricsProps {
    report: TrainingReport;
    dataset: Dataset;
}

export function RegressionMetrics({ report, dataset }: RegressionMetricsProps) {
    const [selectedDataset, setSelectedDataset] = useState<string>('train');

    useEffect(() => {
        // This is a workaround to avoid the issue: Calling setState synchronously within an effect can trigger cascading renders
        setTimeout(() => {
            setSelectedDataset('train');
        }, 0);
    }, [dataset]);

    const supportsRegressionMetrics = 'trainMetrics' in report && report.taskType === 'regression';

    if (!supportsRegressionMetrics) {
        return (
            <div className="w-full h-full p-4 flex items-center justify-center">
                <div className="text-center text-muted-foreground">
                    This model does not support regression metrics.
                </div>
            </div>
        );
    }

    const { trainMetrics, testMetrics = null } = report;

    const hasMetrics = trainMetrics !== null;

    if (!hasMetrics) {
        return (
            <div className="w-full h-full p-4 flex items-center justify-center">
                <div className="text-sm text-center text-muted-foreground">
                    Run training to see regression metrics
                </div>
            </div>
        );
    }

    const metricsData =
        selectedDataset === 'test' && testMetrics !== null ? testMetrics : trainMetrics;

    return (
        <div className="w-full py-4">
            {report.testMetrics && (
                <div className="flex flex-row justify-end px-4">
                    <TrainTestSelector value={selectedDataset} onChange={setSelectedDataset} />
                </div>
            )}

            <MetricsDisplay metrics={metricsData} />
        </div>
    );
}
