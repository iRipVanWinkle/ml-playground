import { Field, Select } from '@/app/shared/ui';
import type { ModelType } from '@/app/models/types';
import type { TaskType } from '@/app/shared/types';
import { getModelRegistry } from '@/app/models/ui-registry';
import { setModelType, useModelType } from '@/app/store';

type ModelTypeProps = {
    taskType: TaskType;
    disabled?: boolean;
};

const modelRegistry = getModelRegistry();

const MODEL_TYPE_INFO = 'The algorithm used to find patterns in the data and make predictions.';

export function ModelTypeSelector({ taskType, disabled }: ModelTypeProps) {
    const modelType = useModelType();

    const modelTypes = modelRegistry.getForTask(taskType);

    const handleChange = (value: string) => {
        setModelType(value as ModelType);
    };

    return (
        <Field label="Model Type" htmlFor="modelType" info={MODEL_TYPE_INFO}>
            <Select disabled={disabled} value={modelType} onValueChange={handleChange}>
                <Select.Trigger
                    className="w-full truncate"
                    id="modelType"
                    data-testid="model-type-select"
                >
                    <Select.Value placeholder="Select Model Type" />
                </Select.Trigger>
                <Select.Content>
                    {modelTypes.map((model) => (
                        <Select.Item key={model.key} value={model.key}>
                            {model.label}
                        </Select.Item>
                    ))}
                </Select.Content>
            </Select>
        </Field>
    );
}
