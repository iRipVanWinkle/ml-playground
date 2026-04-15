import { Card } from '@/app/shared/ui';
import { SettingsRenderer, ModelTypeSelector } from '@/app/features/configure-model';
import { useIsTraining, useTaskType, useNumCategories } from '@/app/store';

export function ModelSection() {
    const taskType = useTaskType();
    const isTraining = useIsTraining();
    const numCategories = useNumCategories();

    return (
        <Card className="gap-5">
            <Card.Header>
                <Card.Title>Model</Card.Title>
            </Card.Header>
            <Card.Content className="grid gap-5">
                <ModelTypeSelector taskType={taskType} disabled={isTraining} />

                <SettingsRenderer
                    taskType={taskType}
                    disabled={isTraining}
                    numCategories={numCategories}
                />
            </Card.Content>
        </Card>
    );
}
