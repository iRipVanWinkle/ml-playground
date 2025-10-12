import { Field, Select } from '@/app/shared/ui';
import { AVAILABLE_BACKENDS, BACKEND_LABELS } from '../constants';
import { useBackendDetection } from '../services';
import { useBackend, type TensorBackend, updateBackend } from '../store';

export type BackendSelectorProps = {
    disabled: boolean;
};

export function BackendSelector({ disabled }: BackendSelectorProps) {
    const { supported = [], current } = useBackendDetection();
    const value = useBackend();

    const availableBackendOptions = [
        {
            value: 'auto',
            label: current ? `Default (${BACKEND_LABELS[current] ?? current})` : 'Default',
        },
        ...AVAILABLE_BACKENDS.filter(({ value }) => supported.includes(value)),
    ];

    return (
        <Field label="TensorFlow Backend" htmlFor="tensorflowBackendSelect">
            <Select
                value={value}
                disabled={disabled}
                onValueChange={(value) => updateBackend(value as TensorBackend)}
            >
                <Select.Trigger
                    id="tensorflowBackendSelect"
                    className="w-full truncate"
                    data-testid="tensorflow-backend-select"
                >
                    <Select.Value placeholder="Select TensorFlow Backend" />
                </Select.Trigger>
                <Select.Content>
                    {availableBackendOptions.map((model) => (
                        <Select.Item key={model.value} value={model.value}>
                            {model.label}
                        </Select.Item>
                    ))}
                </Select.Content>
            </Select>
        </Field>
    );
}
