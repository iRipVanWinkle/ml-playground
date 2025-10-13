import { Field, Select } from '@/app/shared/ui';
import type { ModelType } from '@/app/features/configure-model';
import type { TaskType } from '@/app/shared/types';

const DEFAULT_REGRESSION_MODEL_TYPES = [
    {
        value: 'linear',
        label: 'Linear Regression',
    },
    {
        value: 'neural',
        label: 'Neural Networks',
    },
    {
        value: 'tree',
        label: 'Decision Tree',
    },
];

const DEFAULT_CLASSIFICATION_MODEL_TYPES = [
    {
        value: 'logistic',
        label: 'Logistic Regression',
    },
    {
        value: 'neural',
        label: 'Neural Networks',
    },
    {
        value: 'tree',
        label: 'Decision Tree',
    },
];

type ModelTypeProps = {
    taskType: TaskType;
    value: ModelType;
    onChange: (value: ModelType) => void;
    disabled?: boolean;
};

export function ModelType({ taskType, value, onChange, disabled }: ModelTypeProps) {
    const modelTypes =
        taskType === 'regression'
            ? DEFAULT_REGRESSION_MODEL_TYPES
            : DEFAULT_CLASSIFICATION_MODEL_TYPES;

    return (
        <Field label="Model Type" htmlFor="modelType">
            <Select
                disabled={disabled}
                value={value}
                onValueChange={(value) => onChange(value as ModelType)}
            >
                <Select.Trigger
                    className="w-full truncate"
                    id="modelType"
                    data-testid="model-type-select"
                >
                    <Select.Value placeholder="Select Model Type" />
                </Select.Trigger>
                <Select.Content>
                    {modelTypes.map((model) => (
                        <Select.Item key={model.value} value={model.value}>
                            {model.label}
                        </Select.Item>
                    ))}
                </Select.Content>
            </Select>
        </Field>
    );
}
