import { useMemo, useState } from 'react';
import type { TrainingReport } from '@/app/models/types';
import type { ParametersVisualizationProps } from '@/app/shared/registry';
import type { Dataset, Transformation } from '@/app/shared/types';
import { transposeMatrix, type MatrixLike } from '@/app/shared/helpers';
import { useAllFeatureLabels } from '@/app/shared/visualization/base';
import {
    BinaryParameters,
    MulticlassParameters,
    ImageParameters,
    ViewSelector,
    RawParameters,
} from './components';

export function RegressionParameters({
    report,
    dataset,
    transformations,
}: ParametersVisualizationProps<TrainingReport>) {
    const supportsRegressionParameters = 'theta' in report;

    if (!supportsRegressionParameters) {
        return (
            <div className="w-full h-full p-4 flex items-center justify-center">
                <div className="text-center text-muted-foreground">
                    This model does not support linear parameters.
                </div>
            </div>
        );
    }

    if (report.theta.shape[1] === 0) {
        return (
            <div className="w-full h-full p-4 flex items-center justify-center">
                <div className="text-sm text-center text-muted-foreground">
                    Run training to see learned parameters.
                </div>
            </div>
        );
    }

    const isLinearRegression = report.taskType === 'regression';

    return (
        <RegressionParametersContent
            theta={report.theta}
            dataset={dataset}
            transformations={transformations}
            isLinearRegression={isLinearRegression}
        />
    );
}

interface RegressionParametersContentProps {
    theta: MatrixLike;
    dataset: Dataset;
    transformations: Transformation[];
    isLinearRegression: boolean;
}

function RegressionParametersContent({
    theta,
    dataset,
    transformations,
    isLinearRegression,
}: RegressionParametersContentProps) {
    const [, numClasses] = theta.shape;
    const isBinary = numClasses === 1;
    const isImage = dataset.isImage;

    const [view, setView] = useState('all');

    const allFeatureLabels = useAllFeatureLabels(dataset.headers, transformations);
    const categories = dataset.categories ?? [];

    const colBasedTheta = useMemo(() => transposeMatrix(theta), [theta]);

    // Parse the view value
    const isRawView = view === 'raw';
    const isAllClassesView = view === 'all';
    const selectedClassIndex = !isRawView && !isAllClassesView ? parseInt(view, 10) : undefined;

    const renderContent = () => {
        if (isRawView) {
            return <RawParameters theta={colBasedTheta} />;
        }

        if (isImage) {
            return (
                <ImageParameters
                    theta={colBasedTheta}
                    categories={categories}
                    selectedClassIndex={selectedClassIndex}
                />
            );
        }

        if (isBinary) {
            return (
                <BinaryParameters
                    theta={colBasedTheta}
                    featureLabels={allFeatureLabels}
                    isLinearRegression={isLinearRegression}
                />
            );
        }

        return (
            <MulticlassParameters
                theta={colBasedTheta}
                featureLabels={allFeatureLabels}
                categories={categories}
                selectedClassIndex={selectedClassIndex}
            />
        );
    };

    return (
        <div className="w-full grid grid-cols-1 gap-3">
            <h3 className="mb-4 text-lg font-semibold">Learned Parameters</h3>
            <div className="flex flex-row justify-end">
                <div className="flex flex-row gap-2">
                    <ViewSelector value={view} onChange={setView} classLabels={categories} />
                </div>
            </div>

            {renderContent()}
        </div>
    );
}
