import { useEffect, useRef } from 'react';
import { Card } from '@/app/shared/ui';
import { useTaskType } from '@/app/features/switch-task';
import { useModelSettingsStore } from '@/app/features/configure-model';
import { useDataset, useHasData } from '@/app/features/load-dataset';
import { Controls, useResetTrainingControls } from '@/app/features/control-training';
import {
    ModelDataPlot,
    TabbedVisualizations,
    TrainingMetricsGrid,
    TrainingProgress,
    useResetTrainingReport,
} from '@/app/features/visualize-training';
import type { ModelType } from '@/app/models/types';

export function TrainingSection() {
    const hasData = useHasData();
    const modelSettings = useModelSettingsStore();
    const data = useDataset();
    const taskType = useTaskType();

    const modelTypeRef = useRef<ModelType>(modelSettings.type);
    modelTypeRef.current = modelSettings.type;

    const resetControls = useResetTrainingControls();
    const resetReport = useResetTrainingReport();

    useEffect(() => {
        resetControls();
        resetReport(taskType, modelTypeRef.current);
    }, [data, resetControls, resetReport, taskType]);

    return (
        <Card>
            <Card.Content className="grid gap-4">
                <TrainingProgress
                    controlsComponent={<Controls hasData={hasData} taskType={taskType} />}
                    modelType={modelSettings.type}
                    modelSettings={modelSettings}
                />

                <TrainingMetricsGrid modelType={modelSettings.type} />

                <div className="flex flex-col gap-4">
                    <ModelDataPlot modelType={modelSettings.type} dataset={data} />

                    <TabbedVisualizations dataset={data} modelType={modelSettings.type} />
                </div>
            </Card.Content>
        </Card>
    );
}
