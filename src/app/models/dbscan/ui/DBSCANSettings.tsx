import type { DBSCANSettings as DBSCANSettingsType } from '../types';
import type { ModelSettingsComponentProps } from '@/app/shared/registry';
import { Field, Input } from '@/app/shared/ui';
import { Distance } from '@/app/models/k-means/ui/Distance';
import type { DistanceConfig } from '@/ml/factories';

const EPSILON_INFO =
    'The maximum distance between two points for them to be considered neighbors (core point radius).';
const MIN_POINTS_INFO =
    'Minimum number of points within epsilon radius to qualify as a core point.';

export function DBSCANSettings({
    settings,
    disabled,
    onChange,
}: ModelSettingsComponentProps<DBSCANSettingsType>) {
    return (
        <>
            <div className="grid grid-cols-2 gap-2">
                <Field label="Epsilon (ε)" htmlFor="epsilonInput" info={EPSILON_INFO}>
                    <Input
                        disabled={disabled}
                        id="epsilonInput"
                        data-testid="epsilon-input"
                        type="number"
                        step={0.1}
                        min={0.1}
                        value={settings.epsilon}
                        onChange={(e) => onChange({ epsilon: parseFloat(e.target.value) })}
                    />
                </Field>

                <Field label="Min Points" htmlFor="minPointsInput" info={MIN_POINTS_INFO}>
                    <Input
                        disabled={disabled}
                        id="minPointsInput"
                        data-testid="min-points-input"
                        type="number"
                        step={1}
                        min={1}
                        value={settings.minPoints}
                        onChange={(e) => onChange({ minPoints: parseInt(e.target.value) })}
                    />
                </Field>
            </div>
            <Distance
                settings={settings.distance}
                disabled={disabled}
                onChange={(value: DistanceConfig) => onChange({ distance: value })}
            />
        </>
    );
}
