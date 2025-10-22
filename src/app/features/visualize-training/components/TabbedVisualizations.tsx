import type { Dataset } from '@/app/shared/types';
import type { ModelType } from '@/app/models/types';
import { useModelDefinition } from '@/app/models/ui-registry';
import { useTrainingReport } from '../store';

type TabbedVisualizationsProps = {
    modelType: ModelType;
    dataset: Dataset;
};

export function TabbedVisualizations({ dataset, modelType }: TabbedVisualizationsProps) {
    const report = useTrainingReport();
    const modelDefinition = useModelDefinition(modelType);
    const plots = modelDefinition.visualization.plots;

    return plots?.map(({ component: PlotComponent }, index) => (
        <div key={index} className="h-80 bg-muted rounded-lg flex items-center justify-center">
            <PlotComponent dataset={dataset} report={report} />
        </div>
    ));
}
