import { Field, Select } from '@/app/shared/ui';
import { NORMALIZATION_METHODS } from '../constants';
import { setNormalization, useNormalization } from '@/app/store';

const NORMALIZATION_INFO =
    'Scales features to a standard range. Applied before and after transformation (if present) for numerical stability.';

export type NormalizationSelectorProps = {
    disabled?: boolean;
};

export function NormalizationSelector({ disabled }: NormalizationSelectorProps) {
    const normalization = useNormalization();

    return (
        <Field label="Normalization" htmlFor="normalizationSelect" info={NORMALIZATION_INFO}>
            <Select disabled={disabled} value={normalization ?? 'none'} onValueChange={setNormalization}>
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
