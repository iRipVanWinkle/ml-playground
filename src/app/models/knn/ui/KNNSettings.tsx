import type { KNNSettings as KNNSettingsType, KNNWeights } from '../types';
import type { ModelSettingsComponentProps } from '@/app/shared/registry';
import type { DistanceConfig, DistanceType } from '@/ml/factories';
import { Field, Input, Select } from '@/app/shared/ui';

const K_INFO = 'Number of nearest neighbors to consider for prediction.';
const WEIGHTS_INFO =
    'How neighbor votes are weighted. Uniform gives equal weight to all neighbors; distance weights each neighbor by the inverse of its distance.';
const DISTANCE_INFO = 'Distance metric used to find nearest neighbors.';

const WEIGHTS_OPTIONS = [
    {
        value: 'uniform',
        label: 'Uniform',
        info: 'All k neighbors contribute equally to the prediction.',
    },
    {
        value: 'distance',
        label: 'Distance',
        info: 'Closer neighbors contribute more, weighted by inverse distance.',
    },
];

const DISTANCE_OPTIONS = [
    {
        value: 'euclidean',
        label: 'Euclidean',
        info: 'Straight-line distance between two points.',
    },
    {
        value: 'manhattan',
        label: 'Manhattan',
        info: 'Sum of absolute differences across dimensions.',
    },
    {
        value: 'cosine',
        label: 'Cosine',
        info: 'Cosine of the angle between two vectors.',
    },
];

export function KNNSettings({
    settings,
    disabled,
    onChange,
}: ModelSettingsComponentProps<KNNSettingsType>) {
    const handleChange = (newSettings: Partial<KNNSettingsType>) => {
        onChange({ ...settings, ...newSettings });
    };

    return (
        <>
            <Field label="Neighbors (K)" htmlFor="knnKInput" info={K_INFO}>
                <Input
                    disabled={disabled}
                    placeholder="Number of neighbors"
                    step={1}
                    min={1}
                    type="number"
                    id="knnKInput"
                    data-testid="knn-k-input"
                    value={settings.k}
                    onChange={(e) => handleChange({ k: parseInt(e.target.value) })}
                />
            </Field>

            <Field label="Weights" htmlFor="knnWeightsSelect" info={WEIGHTS_INFO}>
                <Select
                    disabled={disabled}
                    value={settings.weights}
                    onValueChange={(value) => handleChange({ weights: value as KNNWeights })}
                >
                    <Select.Trigger
                        id="knnWeightsSelect"
                        className="w-full truncate"
                        data-testid="knn-weights-select"
                    >
                        <Select.Value placeholder="Select weights" />
                    </Select.Trigger>
                    <Select.Content>
                        {WEIGHTS_OPTIONS.map((opt) => (
                            <Select.Item key={opt.value} value={opt.value} title={opt.info}>
                                {opt.label}
                            </Select.Item>
                        ))}
                    </Select.Content>
                </Select>
            </Field>

            <Field label="Distance" htmlFor="knnDistanceSelect" info={DISTANCE_INFO}>
                <Select
                    disabled={disabled}
                    value={settings.distance.type}
                    onValueChange={(type) =>
                        handleChange({ distance: { type: type as DistanceType } as DistanceConfig })
                    }
                >
                    <Select.Trigger
                        id="knnDistanceSelect"
                        className="w-full truncate"
                        data-testid="knn-distance-select"
                    >
                        <Select.Value placeholder="Select distance metric" />
                    </Select.Trigger>
                    <Select.Content>
                        {DISTANCE_OPTIONS.map((opt) => (
                            <Select.Item key={opt.value} value={opt.value} title={opt.info}>
                                {opt.label}
                            </Select.Item>
                        ))}
                    </Select.Content>
                </Select>
            </Field>
        </>
    );
}
