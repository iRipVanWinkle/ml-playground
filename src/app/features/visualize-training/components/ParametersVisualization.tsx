import type { Dataset, Transformation } from '@/app/shared/types';
import type { ModelSettings, ModelType } from '@/app/models/types';
import { useModelDefinition } from '@/app/models/ui-registry';
import { useTrainingReport } from '../store';
import { Separator } from '@/app/shared/ui';

type ParametersVisualizationProps = {
    modelType: ModelType;
    dataset: Dataset;
    modelSettings: ModelSettings;
    transformations: Transformation[];
};

export function ParametersVisualization({
    dataset,
    modelType,
    modelSettings,
    transformations,
}: ParametersVisualizationProps) {
    const report = useTrainingReport();
    const { visualization } = useModelDefinition(modelType);

    const ParametersVisualizationComponent = visualization.parametersComponent;

    if (!ParametersVisualizationComponent) return null;

    return (
        <>
            <Separator />
            <ParametersVisualizationComponent
                dataset={dataset}
                modelSettings={modelSettings}
                transformations={transformations}
                report={report}
            />
        </>
    );
}
