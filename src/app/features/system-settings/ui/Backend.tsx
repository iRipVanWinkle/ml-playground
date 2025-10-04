import { Field, Select } from '@/app/shared/ui';
import { AVAILABLE_BACKENDS, BACKEND_LABELS } from '../model/constants';
import { useDetectTfjsBackends } from '../model/detect-backends';
import { useBackend } from '../model/hooks';
import type { TensorBackend } from '../model/types';

export type BackendProps = {
    disabled: boolean;
};

export function Backend({ disabled }: BackendProps) {
    const { supported = [], current } = useDetectTfjsBackends();
    const [value, onChange] = useBackend();

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
                onValueChange={(value) => onChange(value as TensorBackend)}
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
