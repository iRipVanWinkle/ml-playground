import { Field } from '@/app/components/ui/field';
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from '@/app/components/ui/select';
import { useSupportedBackends } from '@/app/hooks';
import type { OptionList } from '../types';
import type { TensorBackend } from '@/app/store';

type BackendProps = {
    value: TensorBackend;
    disabled: boolean;
    onChange: (value: TensorBackend) => void;
};

const AVAILABLE_BACKENDS: OptionList = [
    { value: 'webgpu', label: 'WebGPU' },
    { value: 'webgl', label: 'WebGL' },
    { value: 'cpu', label: 'CPU' },
    { value: 'wasm', label: 'WASM' },
];

export function Backend({ value, disabled, onChange }: BackendProps) {
    const supportedBackends = useSupportedBackends();

    const availableBackendOptions = [
        { value: 'auto', label: 'Default' },
        ...AVAILABLE_BACKENDS.filter(({ value }) => supportedBackends.includes(value)),
    ];

    return (
        <Field label="TensorFlow Backend">
            <Select
                value={value}
                disabled={disabled}
                onValueChange={(value) => onChange(value as TensorBackend)}
            >
                <SelectTrigger className="w-full truncate">
                    <SelectValue placeholder="Select TensorFlow Backend" />
                </SelectTrigger>
                <SelectContent>
                    {availableBackendOptions.map((model) => (
                        <SelectItem key={model.value} value={model.value}>
                            {model.label}
                        </SelectItem>
                    ))}
                </SelectContent>
            </Select>
        </Field>
    );
}
