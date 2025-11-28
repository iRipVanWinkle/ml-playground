import type { NaiveBayesSettings as NaiveBayesSettingsType, NaiveBayesVariant } from '../types';
import type { ModelSettingsComponentProps } from '@/app/shared/registry';
import { Field, Select } from '@/app/shared/ui';

export function NaiveBayesSettings({
    settings,
    disabled,
    onChange,
}: ModelSettingsComponentProps<NaiveBayesSettingsType>) {
    const handleChange = (newSettings: Partial<NaiveBayesSettingsType>) => {
        onChange({ ...settings, ...newSettings });
    };

    const variantOptions = [
        { value: 'gaussian', label: 'Gaussian' },
        { value: 'quadratic', label: 'Quadratic' },
    ];

    return (
        <>
            <Field label="Variant" htmlFor="naiveBayesVariant">
                <Select
                    disabled={disabled}
                    value={settings.variant}
                    onValueChange={(variant) =>
                        handleChange({
                            variant: variant as NaiveBayesVariant,
                        })
                    }
                >
                    <Select.Trigger id="naiveBayesVariant" className="w-full truncate">
                        <Select.Value placeholder="Select variant" />
                    </Select.Trigger>
                    <Select.Content>
                        {variantOptions.map((option) => (
                            <Select.Item key={option.value} value={option.value}>
                                {option.label}
                            </Select.Item>
                        ))}
                    </Select.Content>
                </Select>
            </Field>
        </>
    );
}
