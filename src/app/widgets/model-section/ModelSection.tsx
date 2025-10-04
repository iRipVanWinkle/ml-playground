import {
    setModelType,
    updateModelSettings,
    useIsTraining,
    useModelSettings,
    useTaskType,
} from '@/app/store';
import { Card } from '@/app/shared/ui';
import { ModelType } from './components/ModelType';
import { SettingsRenderer } from '@/app/features/configure-model';

export default function ModelSection() {
    const data = useModelSettings();
    const taskType = useTaskType();
    const isTraining = useIsTraining();

    return (
        <Card className="gap-5">
            <Card.Header>
                <Card.Title>Model</Card.Title>
            </Card.Header>
            <Card.Content className="grid gap-5">
                <ModelType
                    taskType={taskType}
                    value={data.type}
                    onChange={(value) => setModelType(value)}
                    disabled={isTraining}
                />

                <SettingsRenderer
                    taskType={taskType}
                    value={data}
                    disabled={isTraining}
                    onChange={(settings) => updateModelSettings(settings)}
                />
            </Card.Content>
        </Card>
    );
}
