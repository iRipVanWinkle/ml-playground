import type { NaiveBayesSettings as NaiveBayesSettingsType, NaiveBayesVariant } from '../types';
import type { ModelSettingsComponentProps } from '@/app/shared/registry';
import { Field, Select } from '@/app/shared/ui';

const VARIANT_INFO =
    'The variant of Naive Bayes to use. Different variants make different assumptions about the data distribution.';

export function NaiveBayesSettings({
    settings,
    disabled,
    onChange,
}: ModelSettingsComponentProps<NaiveBayesSettingsType>) {
    const handleChange = (newSettings: Partial<NaiveBayesSettingsType>) => {
        onChange({ ...settings, ...newSettings });
    };

    const variantOptions = [
        {
            value: 'gaussian',
            label: 'Gaussian',
            info: 'Assumes data follows a bell curve pattern. Good for numbers that can have any value (like height or weight).',
        },
        {
            value: 'quadratic',
            label: 'Quadratic',
            info: 'Assumes each class has its own covariance. Good for discrete counts or categories (like counts).',
        },
    ];

    return (
        <>
            <Field label="Variant" htmlFor="naiveBayesVariant" info={VARIANT_INFO}>
                <Select
                    disabled={disabled}
                    value={settings.variant}
                    onValueChange={(variant) =>
                        handleChange({
                            variant: variant as NaiveBayesVariant,
                        })
                    }
                >
                    <Select.Trigger
                        id="naiveBayesVariant"
                        className="w-full truncate"
                        data-testid="naive-bayes-variant-select"
                    >
                        <Select.Value placeholder="Select variant" />
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
        </>
    );
}
