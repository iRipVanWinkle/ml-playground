import type { TrainingReport } from '@/app/models/types';
import type { Dataset } from '@/app/shared/types';
import { TrainTestSelector } from '@/app/shared/ui';
import { RocCurveMetrics, RocCurvePlot } from './components';
import { hasRocCurveData } from './utils';
import { useState } from 'react';

type RocCurveProps = {
    dataset: Dataset;
    report: TrainingReport;
};

export function RocCurve({ dataset, report }: RocCurveProps) {
    const [selectedDataset, setSelectedDataset] = useState<string>('train');

    const supportsRocCurve = 'trainRocCurve' in report;

    if (!supportsRocCurve) {
        return (
            <div className="w-full h-full p-4 flex items-center justify-center">
                <div className="text-center text-muted-foreground">
                    This model does not support ROC curve.
                </div>
            </div>
        );
    }

    const categories = dataset.categories ?? [];
    const rocCurveData =
        selectedDataset === 'test' && report.testRocCurve
            ? report.testRocCurve
            : report.trainRocCurve;

    const hasData = hasRocCurveData(rocCurveData);

    return (
        <div className="w-full py-4 bg-background">
            {report.testRocCurve && (
                <div className="flex flex-row justify-end">
                    <TrainTestSelector value={selectedDataset} onChange={setSelectedDataset} />
                </div>
            )}
            <RocCurvePlot rocCurveData={rocCurveData} categories={categories} />

            {hasData ? (
                <RocCurveMetrics rocCurveData={rocCurveData} categories={categories} />
            ) : (
                <div className="w-full h-full p-4 flex items-center justify-center bg-muted rounded-lg">
                    <div className="text-center text-muted-foreground">
                        Run training to see AUC scores
                    </div>
                </div>
            )}
        </div>
    );
}
