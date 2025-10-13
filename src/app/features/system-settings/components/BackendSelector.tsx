import { Field, Select } from '@/app/shared/ui';
import { AVAILABLE_BACKENDS, BACKEND_LABELS } from '../constants';
import { useBackendDetection } from '../libs';
import { type TensorBackend, useSystemStore } from '../store';
import { updateBackend } from '../store/actions';

export type BackendSelectorProps = {
    disabled: boolean;
};

const useBackend = () => useSystemStore((state) => state.backend);

export function BackendSelector({ disabled }: BackendSelectorProps) {
    const { supported = [], current } = useBackendDetection();
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
            <Select
                value={backend}
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
