import { Field, Select } from '@/app/shared/ui';
import type { NormalizationMethod } from '../model/types';
import { NORMALIZATIONS } from '../model/constants';
import { useNormalization } from '../model/hooks';

export type NormalizationProps = {
    disabled?: boolean;
};

export function Normalization({ disabled }: NormalizationProps) {
    const [value, onChange] = useNormalization();

    const handleChange = (value: string) => {
        const newValue = value;
        onChange(newValue as NormalizationMethod);
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
                    {NORMALIZATIONS.map((option) => (
                        <Select.Item key={option.value} value={option.value}>
                            {option.label}
                        </Select.Item>
                    ))}
                </Select.Content>
            </Select>
        </Field>
    );
}
