import type { ModelSettingsComponentProps } from '@/app/shared/registry/types/model-definition';
import type { KMeansSettings as KMeansSettingsType } from '../types';
import { Field, Input, InputGroup, Switch } from '@/app/shared/ui';
import { CentroidInitialization } from './CentroidInitialization';
import { Distance } from './Distance';
import type { CentroidInitializationConfig, DistanceConfig } from '@/ml/factories';

const K_INFO = 'The number of clusters to form.';
const MAX_ITERATIONS_INFO = 'Maximum number of iterations for the algorithm to run';
const TOLERANCE_INFO =
    'Tolerance for early stopping based on improvement in inertia between iterations.';

const DEFAULT_TOLERANCE = 0.0001;

export function KMeansSettings({
    settings,
    disabled,
    onChange,
}: ModelSettingsComponentProps<KMeansSettingsType>) {
    const handleChange = (newSettings: Partial<KMeansSettingsType>) => {
        onChange({ ...settings, ...newSettings });
    };

    const handleInputChange = (key: keyof KMeansSettingsType, value: string) => {
        handleChange({ [key]: parseInt(value) });
    };

    const handleToleranceChange = (value: string) => {
        handleChange({ tolerance: value ? parseFloat(value) : undefined });
    };

    const handleToleranceToggle = (enabled: boolean) => {
        handleChange({ tolerance: enabled ? DEFAULT_TOLERANCE : undefined });
    };

    const handleSelectChange = (
        key: keyof KMeansSettingsType,
        value: CentroidInitializationConfig | DistanceConfig,
    ) => {
        handleChange({ [key]: value });
    };

    return (
        <>
            <Field label="Number of Clusters (K)" info={K_INFO}>
                <Input
                    disabled={disabled}
                    placeholder="Number of Clusters (K)"
                    step={1}
                    min={2}
                    type="number"
                    data-testid="num-clusters"
                    value={settings.numClusters}
                    onChange={(e) => handleInputChange('numClusters', e.target.value)}
                />
            </Field>

            <CentroidInitialization
                settings={settings.centroidInitialization}
                disabled={disabled}
                onChange={(value) => handleSelectChange('centroidInitialization', value)}
            />

            <Distance
                settings={settings.distance}
                disabled={disabled}
                onChange={(value) => handleSelectChange('distance', value)}
            />

            <div className="grid grid-cols-2 gap-2">
                <Field label="Max Iterations" info={MAX_ITERATIONS_INFO}>
                    <Input
                        disabled={disabled}
                        placeholder="Max Iterations"
                        step={1}
                        min={2}
                        type="number"
                        data-testid="max-iteration"
                        value={settings.maxIterations}
                        onChange={(e) => handleInputChange('maxIterations', e.target.value)}
                    />
                </Field>

                <Field label="Tolerance" info={TOLERANCE_INFO}>
                    <InputGroup>
                        <InputGroup.Addon>
                            <Switch
                                id="tolerance-enabled"
                                checked={settings.tolerance !== undefined}
                                onCheckedChange={handleToleranceToggle}
                                disabled={disabled}
                                data-testid="tolerance-switch"
                            />
                        </InputGroup.Addon>
                        <InputGroup.Input
                            disabled={disabled || settings.tolerance === undefined}
                            placeholder="Early stopping is off"
                            step={0.0001}
                            min={0}
                            type="number"
                            data-testid="tolerance"
                            value={settings.tolerance ?? ''}
                            onChange={(e) => handleToleranceChange(e.target.value)}
                        />
                    </InputGroup>
                </Field>
            </div>
        </>
    );
}
