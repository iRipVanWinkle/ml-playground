import { Field, Select } from '@/app/shared/ui';
import { AVAILABLE_BACKENDS, BACKEND_LABELS } from '../constants';
import { useBackendDetection } from '../libs';
import { setBackend, useBackend } from '@/app/store';

export type BackendSelectorProps = {
    disabled: boolean;
};

export function BackendSelector({ disabled }: BackendSelectorProps) {
    const { backendInfo } = useBackendDetection();
    const { supported = [], current } = backendInfo;
    const backend = useBackend();

    const availableBackendOptions = [
        {
            value: 'auto',
            label: current ? `Default (${BACKEND_LABELS[current] ?? current})` : 'Default',
        },
        ...AVAILABLE_BACKENDS.filter(({ value }) => supported.includes(value)),
    ];

    return (
        <Field label="TensorFlow Backend" htmlFor="tensorflowBackendSelect">
            <Select value={backend} disabled={disabled} onValueChange={setBackend}>
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
