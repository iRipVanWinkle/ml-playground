import { useIsTraining } from '@/app/features/control-training';
import { Card } from '@/app/shared/ui';
import { BackendSelector, RandomSeedInput } from '@/app/features/configure-system';

export function SystemSettings() {
    const isTraining = useIsTraining();

    return (
        <Card className="gap-5">
            <Card.Header>
                <Card.Title>System Settings</Card.Title>
            </Card.Header>
            <Card.Content className="grid gap-5">
                <BackendSelector disabled={isTraining} />

                <RandomSeedInput disabled={isTraining} />
            </Card.Content>
        </Card>
    );
}
