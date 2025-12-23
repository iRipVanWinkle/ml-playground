import { Field, Select } from '@/app/shared/ui';
import type { CentroidInitializationConfig, CentroidInitializationType } from '@/ml/factories';

const CENTROID_INITIALIZATION_INFO = 'Method to initialize the centroids of the clusters.';

const DEFAULT_CENTROID_INITIALIZATIONS = [
    {
        value: 'random',
        label: 'Random',
        info: 'Selects K random data points as initial centroids.',
    },
    {
        value: 'kmeans++',
        label: 'K-Means++',
        info: 'Chooses initial centroids to be distant from each other, improving convergence speed and clustering quality.',
    },
];

interface CentroidInitializationProps {
    settings: CentroidInitializationConfig;
    disabled: boolean;
    onChange: (config: CentroidInitializationConfig) => void;
}

export function CentroidInitialization({
    settings,
    disabled,
    onChange,
}: CentroidInitializationProps) {
    const handleChange = (type: string) => {
        onChange({ type: type as Exclude<CentroidInitializationType, 'custom'> });
    };

    return (
        <Field label="Centroid Initialization" info={CENTROID_INITIALIZATION_INFO}>
            <Select
                disabled={disabled}
                value={settings.type as string}
                onValueChange={handleChange}
            >
                <Select.Trigger
                    className="w-full truncate"
                    data-testid="centroid-initialization-select"
                >
                    <Select.Value placeholder="Select initialization method" />
                </Select.Trigger>
                <Select.Content>
                    {DEFAULT_CENTROID_INITIALIZATIONS.map((func) => (
                        <Select.Item key={func.value} value={func.value} title={func.info}>
                            {func.label}
                        </Select.Item>
                    ))}
                </Select.Content>
            </Select>
        </Field>
    );
}
