import { useState } from 'react';
import type { TrainingReport } from '@/app/models/types';
import type { Dataset } from '@/app/shared/types';
import { TrainTestSelector } from '@/app/shared/ui';
import { generateLabels } from './utils';
import { MatrixGrid, MetricsDisplay, ViewSelector } from './components';
import type { ConfusionMatrixData } from './types';
import { useMatrixLabels } from './hooks';

interface ConfusionMatrixProps {
    dataset: Dataset;
    report: TrainingReport;
}

export const ConfusionMatrix = ({ dataset, report }: ConfusionMatrixProps) => {
    const [selectedView, setSelectedView] = useState<string>('full');
    const [selectedDataset, setSelectedDataset] = useState<string>('train');

    const supportsConfusionMatrix = 'trainConfusionMatrix' in report;
    const hasConfusionMatrix =
        supportsConfusionMatrix && report.trainConfusionMatrix.matrix.length > 0;

    const categories = dataset.categories ?? [];
    const matrixSize = categories?.length ?? 2;
    const isBinaryClassification = matrixSize === 2;
    const labels = categories ?? generateLabels(matrixSize);

    if (!supportsConfusionMatrix) {
        return (
            <div className="w-full h-full p-4 flex items-center justify-center">
                <div className="text-center text-muted-foreground">
                    The model does not support confusion matrix.
                </div>
            </div>
        );
    }

    if (!hasConfusionMatrix) {
        return (
            <div className="w-full h-full p-4 flex items-center justify-center">
                <div className="text-sm text-center text-muted-foreground">
                    Run training to see confusion matrix
                </div>
            </div>
        );
    }

    const confusionMatrixData =
        selectedDataset === 'test' && report.testConfusionMatrix
            ? report.testConfusionMatrix
            : report.trainConfusionMatrix;

    return (
        <div className="w-full h-full p-4 flex flex-col gap-4">
            <div className="flex flex-row justify-end">
                <div className="flex flex-row gap-2">
                    {!isBinaryClassification && (
                        <ViewSelector
                            value={selectedView}
                            onChange={setSelectedView}
                            labels={labels}
                        />
                    )}

                    {report.testConfusionMatrix && (
                        <TrainTestSelector value={selectedDataset} onChange={setSelectedDataset} />
                    )}
                </div>
            </div>

            <ConfusionMatrixContent
                labels={labels}
                confusionMatrixData={confusionMatrixData}
                selectedView={selectedView}
            />
        </div>
    );
};

interface ConfusionMatrixContentProps {
    labels: string[];
    confusionMatrixData: ConfusionMatrixData;
    selectedView: string;
}

export const ConfusionMatrixContent = ({
    labels,
    confusionMatrixData,
    selectedView,
}: ConfusionMatrixContentProps) => {
    const matrixSize = labels.length;
    const isBinaryClassification = matrixSize === 2;

    const { rowLabels, columnLabels, classLabels } = useMatrixLabels({
        labels,
        selectedView,
        isBinaryClassification,
    });

    const isOneVsRestOrBinary = selectedView !== 'full' || isBinaryClassification;

    let metrics = confusionMatrixData.metrics;
    let displayMatrix = confusionMatrixData.matrix;

    if (confusionMatrixData.perClassMetrics && confusionMatrixData.perClassMatrix) {
        const index = labels.findIndex((label) => label === selectedView);

        if (isOneVsRestOrBinary && index !== -1) {
            metrics = confusionMatrixData.perClassMetrics[index];
            displayMatrix = confusionMatrixData.perClassMatrix[index];
        }
    }

    return (
        <>
            <MatrixGrid
                displayMatrix={displayMatrix}
                rowLabels={rowLabels}
                columnLabels={columnLabels}
                classLabels={classLabels}
            />

            {metrics && <MetricsDisplay metrics={metrics} />}
        </>
    );
};
