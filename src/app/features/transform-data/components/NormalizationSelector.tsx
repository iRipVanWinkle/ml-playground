import { useEffect } from 'react';
import { Field, Select } from '@/app/shared/ui';
import type { TaskType } from '@/app/shared/types';
import { NORMALIZATION_METHODS } from '../constants';
import { type NormalizationMethod, useNormalization } from '../store';
import { resetNormalization, updateNormalization } from '../store/actions';

const NORMALIZATION_INFO =
    'Scales features to a standard range. Applied before and after transformation (if present) for numerical stability.';

export type NormalizationSelectorProps = {
    disabled?: boolean;
    taskType: TaskType;
};

export function NormalizationSelector({ disabled, taskType }: NormalizationSelectorProps) {
    const normalization = useNormalization();

    useEffect(() => {
        resetNormalization();
    }, [taskType]);

    const handleChange = (value: string) => {
        updateNormalization(value as NormalizationMethod);
    };

    return (
        <Field label="Normalization" htmlFor="normalizationSelect" info={NORMALIZATION_INFO}>
            <Select disabled={disabled} value={normalization} onValueChange={handleChange}>
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
