import { Card, CardContent, CardHeader, CardTitle } from '@/app/components/ui/card';
import { updateSystemSettings, useIsTraining, useSystemSettings } from '@/app/store';
import { Backend } from './Backend';
import { RandomSeed } from './RandomSeed';

export function SystemSettings() {
    const isTraining = useIsTraining();
    const settings = useSystemSettings();

    return (
        <Card className="gap-5">
            <CardHeader>
                <CardTitle>System Settings</CardTitle>
            </CardHeader>
            <CardContent className="grid gap-5">
                <Backend
                    value={settings.backend}
                    disabled={isTraining}
                    onChange={(backend) => updateSystemSettings({ backend })}
                />

                <RandomSeed
                    value={settings.randomSeed}
                    disabled={isTraining}
                    onChange={(randomSeed) => updateSystemSettings({ randomSeed })}
                />
            </CardContent>
        </Card>
    );
}
