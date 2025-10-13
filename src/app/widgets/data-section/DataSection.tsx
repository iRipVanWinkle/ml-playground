import { useIsTraining } from '@/app/features/train-model';
import { Card } from '@/app/shared/ui';
import { DataLoader, useNumTrainInputFeatures } from '@/app/features/load-dataset';
import { NormalizationSelector, TransformationBuilder } from '@/app/features/transform-data';
import { useRandomSeed } from '@/app/features/system-settings';
import { useTaskType } from '@/app/features/task-switcher';

export default function DataSection() {
    const isTraining = useIsTraining();
    const taskType = useTaskType();
    const randomSeed = useRandomSeed();
    const numFeatures = useNumTrainInputFeatures();

    return (
        <Card className="gap-5">
            <Card.Header>
                <Card.Title>Dataset</Card.Title>
            </Card.Header>
            <Card.Content className="grid gap-5">
                <DataLoader disabled={isTraining} taskType={taskType} randomSeed={randomSeed} />

                <NormalizationSelector disabled={isTraining} taskType={taskType} />
                <TransformationBuilder
                    disabled={isTraining}
                    numFeatures={numFeatures}
                    taskType={taskType}
                />
            </Card.Content>
        </Card>
    );
}
