import { Card, Separator } from '@/app/shared/ui';
import { useTaskSwitcherStore } from '@/app/features/switch-task';
import { useModelSettingsStore } from '@/app/features/configure-model';
import { useDataset, useDatasetStore, useHasData } from '@/app/features/load-dataset';
import { useTransformationStore, useTransformations } from '@/app/features/transform-data';
import { Controls } from '@/app/features/control-training';
import {
    ModelDataPlot,
    ParametersVisualization,
    TabbedVisualizations,
    TrainingMetricsGrid,
    TrainingProgress,
    useSetTrainingReport,
} from '@/app/features/visualize-training';
import { useSystemStore } from '@/app/features/configure-system';
import type { TrainingSettings } from '@/app/models/types';

function snapshotTrainingSettings(): TrainingSettings {
    const { taskType } = useTaskSwitcherStore.getState();
    const { dataset } = useDatasetStore.getState();
    const modelSettings = useModelSettingsStore.getState();
    const systemSettings = useSystemStore.getState();
    const dataSettings = useTransformationStore.getState();

    return {
        taskType,
        modelSettings,
        systemSettings,
        dataSettings,
        dataset,
    };
}

export function TrainingSection() {
    const hasData = useHasData();
    const modelSettings = useModelSettingsStore();
    const transformations = useTransformations();
    const dataset = useDataset();

    const setTrainingReport = useSetTrainingReport();

    const modelType = modelSettings.type;

    return (
        <Card key={modelType}>
            <Card.Content className="flex flex-col gap-4">
                <TrainingProgress
                    modelType={modelType}
                    modelSettings={modelSettings}
                    dataset={dataset}
                    controlsComponent={
                        <Controls
                            hasData={hasData}
                            snapshotTrainingSettings={snapshotTrainingSettings}
                            setTrainingReport={setTrainingReport}
                        />
                    }
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
