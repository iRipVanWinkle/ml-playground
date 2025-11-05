import type { Dataset } from '@/app/shared/types';
import { Tabs } from '@/app/shared/ui';
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

    if (!plots || plots.length === 0) return null;

    const hasDataset = dataset.trainInputFeatures.length > 0;
    const defaultPlot = plots[0].title;

    return (
        <Tabs defaultValue={defaultPlot} className="w-full">
            <Tabs.List>
                {plots.map(({ title }, index) => (
                    <Tabs.Trigger key={index} value={title} className="px-8 cursor-pointer">
                        {title}
                    </Tabs.Trigger>
                ))}
            </Tabs.List>
            {plots.map(({ title, component: PlotComponent }, index) => (
                <Tabs.Content key={index} value={title}>
                    <div className="min-h-80 bg-muted rounded-lg flex items-center justify-center">
                        {hasDataset ? <PlotComponent dataset={dataset} report={report} /> : null}
                    </div>
                </Tabs.Content>
            ))}
        </Tabs>
    );
}
