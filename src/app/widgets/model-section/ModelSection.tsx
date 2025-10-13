import { useEffect } from 'react';
import { Card } from '@/app/shared/ui';
import { useTaskType } from '@/app/features/task-switcher';
import { useIsTraining } from '@/app/features/train-model';
import {
    SettingsRenderer,
    useModelSettingsStore,
    setModelType,
    updateModelSettings,
} from '@/app/features/configure-model';
import { useNumCategories } from '@/app/features/load-dataset';
import { ModelType } from './components/ModelType';

export default function ModelSection() {
    const data = useModelSettingsStore();
    const taskType = useTaskType();
    const isTraining = useIsTraining();
    const numCategories = useNumCategories();

    useEffect(() => {
        const modelType = taskType === 'regression' ? 'linear' : 'logistic';
        setModelType(modelType, taskType);
    }, [taskType]);

    return (
        <Card className="gap-5">
            <Card.Header>
                <Card.Title>Model</Card.Title>
            </Card.Header>
            <Card.Content className="grid gap-5">
                <ModelType
                    taskType={taskType}
                    value={data.type}
                    onChange={(value) => setModelType(value, taskType)}
                    disabled={isTraining}
                />

                <SettingsRenderer
                    taskType={taskType}
                    value={data}
                    disabled={isTraining}
                    numCategories={numCategories}
                    onChange={(settings) => updateModelSettings(settings)}
                />
            </Card.Content>
        </Card>
    );
}
