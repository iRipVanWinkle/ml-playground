import { Card, Separator } from '@/app/shared/ui';
import { useTaskType } from '@/app/features/switch-task';
import { useModelSettingsStore } from '@/app/features/configure-model';
import { useDataset, useHasData } from '@/app/features/load-dataset';
import { Controls } from '@/app/features/control-training';
import {
    ModelDataPlot,
    TabbedVisualizations,
    TrainingMetricsGrid,
    TrainingProgress,
} from '@/app/features/visualize-training';

export function TrainingSection() {
    const hasData = useHasData();
    const modelSettings = useModelSettingsStore();
    const data = useDataset();
    const taskType = useTaskType();

    const modelType = modelSettings.type;

    return (
        <Card key={modelType}>
            <Card.Content className="flex flex-col gap-4">
                <TrainingProgress
                    controlsComponent={<Controls hasData={hasData} taskType={taskType} />}
                    modelType={modelType}
                    modelSettings={modelSettings}
                />

                <TrainingMetricsGrid modelType={modelType} />

                <div className="flex flex-col gap-4">
                    <ModelDataPlot modelType={modelType} dataset={data} />

                    <Separator />

                    <TabbedVisualizations dataset={data} modelType={modelType} />
                </div>
            </Card.Content>
        </Card>
    );
}
