import { Field, Select } from '@/app/shared/ui';
import type { DistanceConfig, DistanceType } from '@/ml/factories';

const DISTANCE_INFO = 'Metric used to measure the distance between data points and centroids.';

const DEFAULT_DISTANCE_FUNCTIONS = [
    {
        value: 'euclidean',
        label: 'Euclidean',
        info: 'Standard distance metric measuring straight-line distance between points.',
    },
    {
        value: 'manhattan',
        label: 'Manhattan',
        info: 'Measures distance as the sum of absolute differences across dimensions.',
    },
    {
        value: 'cosine',
        label: 'Cosine',
        info: 'Measures the cosine of the angle between two vectors, useful for high-dimensional data.',
    },
];

interface DistanceProps {
    settings: DistanceConfig;
    disabled: boolean;
    onChange: (config: DistanceConfig) => void;
}

export function Distance({ settings, disabled, onChange }: DistanceProps) {
    const handleChange = (type: string) => {
        onChange({ type: type as DistanceType });
    };

    return (
        <Field label="Distance" htmlFor="distanceSelect" info={DISTANCE_INFO}>
            <Select
                disabled={disabled}
                value={settings.type as string}
                onValueChange={handleChange}
            >
                <Select.Trigger
                    id="distanceSelect"
                    className="w-full truncate"
                    data-testid="distance-select"
                >
                    <Select.Value placeholder="Select distance metric" />
                </Select.Trigger>
                <Select.Content>
                    {DEFAULT_DISTANCE_FUNCTIONS.map((func) => (
                        <Select.Item key={func.value} value={func.value} title={func.info}>
                            {func.label}
                        </Select.Item>
                    ))}
                </Select.Content>
            </Select>
        </Field>
    );
}
