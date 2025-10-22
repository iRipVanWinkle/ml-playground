import type { Dataset } from '@/app/shared/types';
import type { ModelType } from '@/app/models/types';
import { useModelDefinition } from '@/app/models/ui-registry';
import { useTrainingReport } from '../store';

type ModelDataPlotProps = {
    modelType: ModelType;
    dataset: Dataset;
};

export function ModelDataPlot({ modelType, dataset }: ModelDataPlotProps) {
    const report = useTrainingReport();
    const modelDefinition = useModelDefinition(modelType);
    const ModelDataPlotComponent = modelDefinition.visualization.modelDataPlotComponent;

    return (
        <div className="min-h-120 bg-muted rounded-lg flex items-center justify-center posotion-relative">
            <ModelDataPlotComponent dataset={dataset} report={report} />
        </div>
    );
}
