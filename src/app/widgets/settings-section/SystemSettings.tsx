import { useIsTraining } from '@/app/store';
import { Card } from '@/app/shared/ui';
import { Backend, RandomSeed } from '@/app/features/system-settings';

export default function SystemSettings() {
    const isTraining = useIsTraining();

    return (
        <Card className="gap-5">
            <Card.Header>
                <Card.Title>System Settings</Card.Title>
            </Card.Header>
            <Card.Content className="grid gap-5">
                <Backend disabled={isTraining} />

                <RandomSeed disabled={isTraining} />
            </Card.Content>
        </Card>
    );
}
