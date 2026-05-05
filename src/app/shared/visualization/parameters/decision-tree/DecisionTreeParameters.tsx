import type { ParametersVisualizationProps } from '@/app/shared/registry';
import type { TrainingReport } from '@/app/models/types';
import { DecisionTreeVisualizer } from './DecisionTreeVisualizer';

export function DecisionTreeParameters({
    report,
    dataset,
}: ParametersVisualizationProps<TrainingReport>) {
    const supportsTreeParameters = report.type === 'tree' || report.type === 'isolation-forest';
    const featureLabels = dataset.headers.slice(1);

    if (!supportsTreeParameters) {
        return (
            <div className="w-full h-full p-4 flex items-center justify-center">
                <div className="text-center text-muted-foreground">
                    This model does not support Decision Tree parameters.
                </div>
            </div>
        );
    }

    if (report.params.length === 0) {
        return (
            <div className="w-full h-full p-4 flex items-center justify-center">
                <div className="text-center text-muted-foreground">
                    Run training to see learned trees.
                </div>
            </div>
        );
    }

    return (
        <div className="w-full h-full flex flex-col">
            <DecisionTreeVisualizer
                trees={report.params}
                featureLabels={featureLabels}
                categories={dataset.categories}
            />
        </div>
    );
}
