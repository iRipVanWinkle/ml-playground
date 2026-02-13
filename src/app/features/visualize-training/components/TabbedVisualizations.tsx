import type { Dataset } from '@/app/shared/types';
import { EnhancedTabs } from '@/app/shared/ui';
import type { ModelType } from '@/app/models/types';
import { useModelDefinition } from '@/app/models/ui-registry';
import { useTrainingReport } from '../store';
import type { PlotVisualization } from '@/app/shared/registry';

type TabbedVisualizationsProps = {
    modelType: ModelType;
    dataset: Dataset;
};

export function TabbedVisualizations({ dataset, modelType }: TabbedVisualizationsProps) {
    const report = useTrainingReport();
    const { visualization } = useModelDefinition(modelType);

    let plots: Array<PlotVisualization<ModelType>> = [];

    if (visualization.plots) {
        plots =
            typeof visualization.plots === 'function'
                ? visualization.plots(report.taskType)
                : visualization.plots;
    }

    if (!plots || plots.length === 0) return null;

    const hasDataset = dataset.trainInputFeatures.length > 0;
    const defaultPlot = plots[0].title;

    return (
        <EnhancedTabs defaultValue={defaultPlot} scrollable>
            <EnhancedTabs.List>
                {plots.map(({ title }, index) => (
                    <EnhancedTabs.Trigger key={index} value={title} className="px-8 cursor-pointer">
                        {title}
                    </EnhancedTabs.Trigger>
                ))}
            </EnhancedTabs.List>
            {plots.map(({ title, component: PlotComponent }, index) => (
                <EnhancedTabs.Content key={index} value={title}>
                    <div className="min-h-40 bg-muted/30 rounded-lg flex items-center justify-center">
                        {hasDataset ? <PlotComponent dataset={dataset} report={report} /> : null}
                    </div>
                </EnhancedTabs.Content>
            ))}
        </EnhancedTabs>
    );
}
