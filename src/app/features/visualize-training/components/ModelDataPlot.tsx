import type { Dataset } from '@/app/shared/types';
import type { ModelSettings, ModelType } from '@/app/models/types';
import { useModelDefinition } from '@/app/models/ui-registry';
import { useTrainingReport, useUserExample } from '@/app/store';

type ModelDataPlotProps = {
    modelType: ModelType;
    modelSettings: ModelSettings;
    dataset: Dataset;
};

export function ModelDataPlot({ modelType, modelSettings, dataset }: ModelDataPlotProps) {
    const report = useTrainingReport();
    const userExample = useUserExample();
    const modelDefinition = useModelDefinition(modelType);
    const ModelDataPlotComponent = modelDefinition.visualization.modelDataPlotComponent;

    return (
        <div className="min-h-120 bg-muted rounded-lg grid place-items-center">
            <ModelDataPlotComponent
                dataset={dataset}
                report={report}
                modelSettings={modelSettings}
                userExample={userExample}
            />
        </div>
    );
}
