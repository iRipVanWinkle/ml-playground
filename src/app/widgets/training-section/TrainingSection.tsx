import { Card, Separator } from '@/app/shared/ui';
import { useDataset, useHasData, useModelSettings, useTransformations } from '@/app/store';
import { Controls } from '@/app/features/control-training';
import {
    ModelDataPlot,
    ParametersVisualization,
    TabbedVisualizations,
    TrainingMetricsGrid,
    TrainingProgress,
} from '@/app/features/visualize-training';

export function TrainingSection() {
    const hasData = useHasData();
    const modelSettings = useModelSettings();
    const transformations = useTransformations();
    const dataset = useDataset();

    const modelType = modelSettings.type;

    return (
        <Card key={modelType}>
            <Card.Content className="flex flex-col gap-4">
                <TrainingProgress
                    modelType={modelType}
                    modelSettings={modelSettings}
                    dataset={dataset}
                    controlsComponent={<Controls hasData={hasData} />}
                />

                <TrainingMetricsGrid modelType={modelType} />

                <div className="flex flex-col gap-4">
                    <ModelDataPlot
                        modelType={modelType}
                        dataset={dataset}
                        modelSettings={modelSettings}
                    />

                    <Separator />

                    <TabbedVisualizations modelType={modelType} dataset={dataset} />

                    <ParametersVisualization
                        modelType={modelType}
                        modelSettings={modelSettings}
                        transformations={transformations}
                        dataset={dataset}
                    />
                </div>
            </Card.Content>
        </Card>
    );
}
