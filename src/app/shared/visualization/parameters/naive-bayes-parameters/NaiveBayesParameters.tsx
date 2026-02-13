import { useState } from 'react';
import type { TrainingReport } from '@/app/models/types';
import type { ParametersVisualizationProps } from '@/app/shared/registry';
import type { Dataset, Transformation } from '@/app/shared/types';
import { useAllFeatureLabels } from './hooks';
import { ImageParameters, ViewSelector, RawParameters, TabularParameters } from './components';
import type { NaiveBayesParams } from '@/ml/types';

export function NaiveBayesParameters({
    report,
    dataset,
    transformations,
}: ParametersVisualizationProps<TrainingReport>) {
    const supportsNaiveBayesParameters = report.type === 'naive-bayes';
    if (!supportsNaiveBayesParameters) {
        return (
            <div className="w-full h-full p-4 flex items-center justify-center">
                <div className="text-center text-muted-foreground">
                    This model does not support Naive Bayes parameters.
                </div>
            </div>
        );
    }

    if (report.params === undefined) {
        return (
            <div className="w-full h-full p-4 flex items-center justify-center">
                <div className="text-center text-muted-foreground">
                    Run training to see learned parameters.
                </div>
            </div>
        );
    }

    return (
        <NaiveBayesParametersContent
            params={report.params}
            dataset={dataset}
            transformations={transformations}
        />
    );
}

interface NaiveBayesParametersContentProps {
    params: NaiveBayesParams;
    dataset: Dataset;
    transformations: Transformation[];
}

function NaiveBayesParametersContent({
    params,
    dataset,
    transformations,
}: NaiveBayesParametersContentProps) {
    const isImage = dataset.isImage;

    const [view, setView] = useState('all');

    const allFeatureLabels = useAllFeatureLabels(dataset.headers, transformations);
    const categories = dataset.categories ?? [];

    const isRawView = view === 'raw';
    const isAllClassesView = view === 'all';
    const selectedClassIndex = !isRawView && !isAllClassesView ? parseInt(view, 10) : undefined;

    const renderContent = () => {
        if (isRawView) {
            return <RawParameters params={params} categories={categories} />;
        }

        if (isImage) {
            return (
                <ImageParameters
                    params={params}
                    categories={categories}
                    selectedClassIndex={selectedClassIndex}
                />
            );
        }

        return (
            <TabularParameters
                params={params}
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
