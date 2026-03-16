import type {
    HierarchicalClusteringSettings as HierarchicalClusteringSettingsType,
    HierarchicalMethod,
    Linkage,
} from '../types';
import type { ModelSettingsComponentProps } from '@/app/shared/registry';
import { Field, Input, Label, RadioGroup, Select } from '@/app/shared/ui';
import { Distance } from '@/app/models/k-means/ui/Distance';
import type { DistanceConfig } from '@/ml/factories';

const CLUSTERING_METHOD_INFO =
    'Divisive splits a single cluster top-down; Agglomerative merges points bottom-up.';
const NUM_CLUSTERS_INFO = 'The final number of clusters to produce.';
const BISECT_ITERATIONS_INFO =
    'Maximum number of k-means iterations run during each bisection step.';
const BISECT_RESTARTS_INFO =
    'Number of random-seed restarts per bisection step. The split with the lowest SSE is kept.';
const LINKAGE_INFO = 'Linkage method for agglomerative clustering.';
const LINKAGE_OPTIONS: { value: Linkage; label: string; info: string }[] = [
    { value: 'ward', label: 'Ward', info: 'Minimizes the variance of the clusters being merged.' },
    {
        value: 'complete',
        label: 'Complete',
        info: 'Uses the maximum distance between observations of the two sets.',
    },
    {
        value: 'average',
        label: 'Average',
        info: 'Uses the average of the distances of each observation of the two sets.',
    },
    {
        value: 'single',
        label: 'Single',
        info: 'Uses the minimum of the distances between all observations of the two sets.',
    },
];

const DEFAULT_STRATEGIES = [
    { value: 'divisive', label: 'Divisive (Top-Down)', info: 'Divisive clustering algorithm.' },
    {
        value: 'agglomerative',
        label: 'Agglomerative (Bottom-Up)',
        info: 'Agglomerative clustering algorithm.',
    },
];

export function HierarchicalClusteringSettings({
    settings,
    disabled,
    onChange,
}: ModelSettingsComponentProps<HierarchicalClusteringSettingsType>) {
    return (
        <>
            <Field label="Clustering Method" info={CLUSTERING_METHOD_INFO}>
                <RadioGroup
                    value={settings.method}
                    onValueChange={(value) => onChange({ method: value as HierarchicalMethod })}
                    disabled={disabled}
                    className="w-full justify-between gap-3 rounded-lg border p-3 transition-colors"
                    aria-label="Model Variant"
                >
                    {DEFAULT_STRATEGIES.map((model) => {
                        return (
                            <div className="flex items-center space-x-2" key={model.value}>
                                <RadioGroup.Item value={model.value} id={model.value} />
                                <Label
                                    className="font-normal"
                                    htmlFor={model.value}
                                    title={model.info}
                                >
                                    {model.label}
                                </Label>
                            </div>
                        );
                    })}
                </RadioGroup>
            </Field>

            <Field label="Number of Clusters" htmlFor="numClustersInput" info={NUM_CLUSTERS_INFO}>
                <Input
                    disabled={disabled}
                    id="numClustersInput"
                    data-testid="num-clusters-input"
                    type="number"
                    step={1}
                    min={2}
                    value={settings.numClusters}
                    onChange={(e) => onChange({ numClusters: parseInt(e.target.value) || 2 })}
                />
            </Field>

            <Distance
                settings={settings.distance}
                disabled={disabled}
                onChange={(value: DistanceConfig) => onChange({ distance: value })}
            />

            {settings.method === 'divisive' && (
                <div className="grid grid-cols-2 gap-2">
                    <Field
                        label="Bisect Iterations"
                        htmlFor="bisectIterationsInput"
                        info={BISECT_ITERATIONS_INFO}
                    >
                        <Input
                            disabled={disabled}
                            id="bisectIterationsInput"
                            data-testid="bisect-iterations-input"
                            type="number"
                            step={1}
                            min={1}
                            value={settings.bisectIterations}
                            onChange={(e) =>
                                onChange({ bisectIterations: parseInt(e.target.value) || 20 })
                            }
                        />
                    </Field>

                    <Field
                        label="Bisect Restarts"
                        htmlFor="bisectRestartsInput"
                        info={BISECT_RESTARTS_INFO}
                    >
                        <Input
                            disabled={disabled}
                            id="bisectRestartsInput"
                            data-testid="bisect-restarts-input"
                            type="number"
                            step={1}
                            min={1}
                            value={settings.bisectRestarts}
                            onChange={(e) =>
                                onChange({ bisectRestarts: parseInt(e.target.value) || 3 })
                            }
                        />
                    </Field>
                </div>
            )}

            {settings.method === 'agglomerative' && (
                <Field label="Linkage" htmlFor="linkageSelect" info={LINKAGE_INFO}>
                    <Select
                        disabled={disabled}
                        value={settings.linkage}
                        onValueChange={(value) => onChange({ linkage: value as Linkage })}
                    >
                        <Select.Trigger
                            className="w-full truncate"
                            id="linkageSelect"
                            data-testid="linkage-select"
                        >
                            <Select.Value placeholder="Select linkage" />
                        </Select.Trigger>
                        <Select.Content>
                            {LINKAGE_OPTIONS.map((opt) => (
                                <Select.Item key={opt.value} value={opt.value} title={opt.info}>
                                    {opt.label}
                                </Select.Item>
                            ))}
                        </Select.Content>
                    </Select>
                </Field>
            )}
        </>
    );
}
