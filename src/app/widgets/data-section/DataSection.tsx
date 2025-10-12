import { useIsTraining } from '@/app/store';
import { useNumTrainInputFeatures } from '@/app/features/load-dataset/store/hooks';
import { Card } from '@/app/shared/ui';
import { DataLoader } from '@/app/features/load-dataset';
import { NormalizationSelector, TransformationBuilder } from '@/app/features/transform-data';

export default function DataSection() {
    const numFeatures = useNumTrainInputFeatures();
    const isTraining = useIsTraining();

    return (
        <Card className="gap-5">
            <Card.Header>
                <Card.Title>Dataset</Card.Title>
            </Card.Header>
            <Card.Content className="grid gap-5">
                <DataLoader disabled={isTraining} />

                <NormalizationSelector disabled={isTraining} />
                <TransformationBuilder disabled={isTraining} numFeatures={numFeatures} />
            </Card.Content>
        </Card>
    );
}
