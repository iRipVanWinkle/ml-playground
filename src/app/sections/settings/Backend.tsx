import { Field } from '@/app/components/ui/field';
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from '@/app/components/ui/select';
import { useDetectTfjsBackends } from '@/app/hooks';
import type { OptionList } from '../types';
import type { TensorBackend } from '@/app/store';

type BackendProps = {
    value: TensorBackend;
    disabled: boolean;
    onChange: (value: TensorBackend) => void;
};

const BACKEND_LABELS: Record<string, string> = {
    webgpu: 'WebGPU',
    webgl: 'WebGL',
    cpu: 'CPU',
    wasm: 'WASM',
};

const AVAILABLE_BACKENDS: OptionList = [
    { value: 'webgpu', label: BACKEND_LABELS['webgpu'] },
    { value: 'webgl', label: BACKEND_LABELS['webgl'] },
    { value: 'cpu', label: BACKEND_LABELS['cpu'] },
    { value: 'wasm', label: BACKEND_LABELS['wasm'] },
];

export function Backend({ value, disabled, onChange }: BackendProps) {
    const { supported = [], current } = useDetectTfjsBackends();

    const availableBackendOptions = [
        { value: 'auto', label: current ? `Default (${BACKEND_LABELS[current] ?? current})` : 'Default' },
        ...AVAILABLE_BACKENDS.filter(({ value }) => supported.includes(value)),
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
