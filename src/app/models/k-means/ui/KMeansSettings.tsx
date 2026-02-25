import type { CentroidInitializationConfig, DistanceConfig } from '@/ml/factories';
import type { ModelSettingsComponentProps } from '@/app/shared/registry/types/model-definition';
import type { KMeansSettings as KMeansSettingsType } from '../types';
import { Field, Input, ToleranceInput } from '@/app/shared/ui';
import { CentroidInitialization } from './CentroidInitialization';
import { Distance } from './Distance';

const K_INFO = 'The number of clusters to form.';
const MAX_ITERATIONS_INFO = 'Maximum number of iterations for the algorithm to run';
const TOLERANCE_INFO =
    'Tolerance for early stopping based on improvement in inertia between iterations.';

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

    const handleSelectChange = (
        key: keyof KMeansSettingsType,
        value: CentroidInitializationConfig | DistanceConfig,
    ) => {
        handleChange({ [key]: value });
    };

    return (
        <>
            <Field label="Number of Clusters (K)" htmlFor="numClustersInput" info={K_INFO}>
                <Input
                    disabled={disabled}
                    placeholder="Number of Clusters (K)"
                    step={1}
                    min={2}
                    type="number"
                    id="numClustersInput"
                    data-testid="num-clusters-input"
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
                <Field
                    label="Max Iterations"
                    htmlFor="maxIterationsInput"
                    info={MAX_ITERATIONS_INFO}
                >
                    <Input
                        disabled={disabled}
                        placeholder="Max Iterations"
                        step={1}
                        min={2}
                        type="number"
                        id="maxIterationsInput"
                        data-testid="max-iterations-input"
                        value={settings.maxIterations}
                        onChange={(e) => handleInputChange('maxIterations', e.target.value)}
                    />
                </Field>

                <Field label="Tolerance" htmlFor="toleranceInput" info={TOLERANCE_INFO}>
                    <ToleranceInput
                        id="toleranceInput"
                        value={settings.tolerance}
                        disabled={disabled}
                        onChange={(value) => handleChange({ tolerance: value })}
                    />
                </Field>
            </div>
        </>
    );
}
