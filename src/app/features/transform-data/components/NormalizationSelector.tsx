import { Field, Select } from '@/app/shared/ui';
import { NORMALIZATION_METHODS } from '../constants';
import { useNormalization, type NormalizationMethod } from '../store';
import { updateNormalization } from '../store/actions';

export type NormalizationSelectorProps = {
    disabled?: boolean;
};

export function NormalizationSelector({ disabled }: NormalizationSelectorProps) {
    const value = useNormalization();

    const handleChange = (value: string) => {
        updateNormalization(value as NormalizationMethod);
    };

    return (
        <Field label="Normalization" htmlFor="normalizationSelect">
            <Select disabled={disabled} value={value} onValueChange={handleChange}>
                <Select.Trigger
                    id="normalizationSelect"
                    className="w-50"
                    data-testid="normalization-select"
                >
                    <Select.Value placeholder="Select normalization" />
                </Select.Trigger>
                <Select.Content>
                    {NORMALIZATION_METHODS.map((option) => (
                        <Select.Item key={option.value} value={option.value}>
                            {option.label}
                        </Select.Item>
                    ))}
                </Select.Content>
            </Select>
        </Field>
    );
}
