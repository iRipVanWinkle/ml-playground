import { Card } from '@/app/shared/ui';
import { DataLoader } from '@/app/features/load-dataset';
import { NormalizationSelector, TransformationBuilder } from '@/app/features/transform-data';
import { useIsTraining, useTaskType, useRandomSeed, useNumTrainInputFeatures } from '@/app/store';

export function DataSection() {
    const isTraining = useIsTraining();
    const taskType = useTaskType();
    const randomSeed = useRandomSeed();
    const numFeatures = useNumTrainInputFeatures();

    return (
        <Card className="gap-5" key={taskType}>
            <Card.Header>
                <Card.Title>Dataset</Card.Title>
            </Card.Header>
            <Card.Content className="grid gap-5">
                <DataLoader disabled={isTraining} taskType={taskType} randomSeed={randomSeed} />

                <NormalizationSelector disabled={isTraining} />
                <TransformationBuilder disabled={isTraining} numFeatures={numFeatures} />
            </Card.Content>
        </Card>
    );
}
