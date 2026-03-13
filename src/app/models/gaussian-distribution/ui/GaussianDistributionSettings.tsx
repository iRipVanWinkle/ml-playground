import type {
    GaussianDistributionSettings as GaussianDistributionSettingsType,
    GaussianDistributionVariant,
} from '../types';
import type { ModelSettingsComponentProps } from '@/app/shared/registry';
import { Field, Input, Select } from '@/app/shared/ui';

const VARIANT_INFO =
    'The covariance structure to use. Diagonal assumes independent features; Full captures feature correlations.';
const THRESHOLD_INFO =
    'Probability density threshold below which a sample is flagged as an anomaly.';
const SMOOTHING_INFO = 'Small value added to the variance to prevent numerical instability.';

export function GaussianDistributionSettings({
    settings,
    disabled,
    onChange,
}: ModelSettingsComponentProps<GaussianDistributionSettingsType>) {
    const handleChange = (newSettings: Partial<GaussianDistributionSettingsType>) => {
        onChange({ ...settings, ...newSettings });
    };

    const variantOptions = [
        {
            value: 'diagonal',
            label: 'Diagonal',
            info: 'Assumes features are independent. Faster and more robust with limited data.',
        },
        {
            value: 'full',
            label: 'Full',
            info: 'Models full feature covariance. Better when features are correlated.',
        },
    ];

    return (
        <>
            <Field label="Covariance Type" htmlFor="gaussianVariant" info={VARIANT_INFO}>
                <Select
                    disabled={disabled}
                    value={settings.variant}
                    onValueChange={(variant) =>
                        handleChange({ variant: variant as GaussianDistributionVariant })
                    }
                >
                    <Select.Trigger
                        id="gaussianVariant"
                        className="w-full truncate"
                        data-testid="gaussian-variant-select"
                    >
                        <Select.Value placeholder="Select covariance type" />
                    </Select.Trigger>
                    <Select.Content>
                        {variantOptions.map((option) => (
                            <Select.Item
                                key={option.value}
                                value={option.value}
                                title={option.info}
                            >
                                {option.label}
                            </Select.Item>
                        ))}
                    </Select.Content>
                </Select>
            </Field>

            <Field label="Anomaly Threshold" htmlFor="gaussianThreshold" info={THRESHOLD_INFO}>
                <Input
                    disabled={disabled}
                    id="gaussianThreshold"
                    data-testid="gaussian-threshold-input"
                    type="number"
                    step="any"
                    min={0}
                    value={settings.threshold}
                    onChange={(e) => handleChange({ threshold: parseFloat(e.target.value) || 0 })}
                />
            </Field>

            <Field label="Variance Smoothing" htmlFor="gaussianSmoothing" info={SMOOTHING_INFO}>
                <Input
                    disabled={disabled}
                    id="gaussianSmoothing"
                    data-testid="gaussian-smoothing-input"
                    type="number"
                    step="any"
                    min={0}
                    value={settings.varianceSmoothing}
                    onChange={(e) =>
                        handleChange({ varianceSmoothing: parseFloat(e.target.value) || 0 })
                    }
                />
            </Field>
        </>
    );
}
