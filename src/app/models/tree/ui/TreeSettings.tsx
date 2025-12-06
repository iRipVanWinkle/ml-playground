import type { TreeSettings as TreeSettingsType, TreeModelVariant } from '../types';
import { Field, Input, Label, RadioGroup } from '@/app/shared/ui';
import Criterion from './Criterion';
import type { ModelSettingsComponentProps } from '@/app/shared/registry';

const MODEL_VARIANT_INFO = 'The variant of the model to use.';
const MAX_DEPTH_INFO = 'Limits how deep the tree can grow.';
const MIN_SAMPLES_SPLIT_INFO = 'Minimum number of samples needed before the tree can split a node.';
const MIN_SAMPLES_LEAF_INFO = 'Minimum number of samples required in any leaf node.';
const ESTIMATORS_INFO = 'The number of decision trees to use in the ensemble.';
const MAX_FEATURES_INFO =
    'The maximum number of features to consider when looking for the best split.';
const NUM_RANDOM_THRESHOLDS_INFO =
    'The number of random thresholds to consider when looking for the best split.';

const DEFAULT_MODEL_VARIANTS = [
    {
        value: 'decision',
        label: 'Single Decision Tree',
        info: 'Simple model that makes decisions based on a series of binary splits.',
    },
    {
        value: 'bagging',
        label: 'Bagging',
        info: 'Creates multiple trees using random samples of data. Averages their predictions for better results.',
    },
    {
        value: 'forest',
        label: 'Random Forest',
        info: 'Creates multiple trees using random samples of data and random features. Combines their predictions.',
    },
    {
        value: 'extra',
        label: 'Extra Trees',
        info: 'Creates multiple trees using random samples, random features, and random split points. Even more randomness than Random Forest.',
    },
];

export function TreeSettings({
    taskType,
    settings,
    disabled,
    onChange,
}: ModelSettingsComponentProps<TreeSettingsType>) {
    const {
        criterion,
        modelVariant,
        maxDepth,
        minSamplesSplit,
        minSamplesLeaf,
        maxFeatures,
        numRandomThresholds,
        estimators,
    } = settings;

    const handleChange = (newSettings: Partial<TreeSettingsType>) => {
        onChange({ ...settings, ...newSettings });
    };

    const handleModelVariantChange = (value: string) => {
        handleChange({ modelVariant: value as TreeModelVariant });
    };

    const handleInputChange = (key: keyof TreeSettingsType, value: string) => {
        handleChange({ [key]: parseInt(value) });
    };

    const needsEstimators =
        modelVariant === 'bagging' || modelVariant === 'forest' || modelVariant === 'extra';
    const needsMaxFeatures = modelVariant === 'forest' || modelVariant === 'extra';
    const needsRandomThresholds = modelVariant === 'extra';

    return (
        <>
            <Field label="Model Variant" info={MODEL_VARIANT_INFO}>
                <RadioGroup
                    value={modelVariant}
                    onValueChange={handleModelVariantChange}
                    disabled={disabled}
                    className="w-full justify-between gap-3 rounded-lg border p-3 transition-colors"
                >
                    {DEFAULT_MODEL_VARIANTS.map((model) => {
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
            <Criterion
                taskType={taskType}
                criterion={criterion}
                disabled={disabled}
                onChange={(criterion) => onChange({ ...settings, criterion })}
            />
            <Field label="Max Depth" info={MAX_DEPTH_INFO}>
                <Input
                    disabled={disabled}
                    placeholder="Max Depth"
                    step={1}
                    min={2}
                    type="number"
                    data-testid="max-depth-input"
                    value={maxDepth}
                    onChange={(e) => handleInputChange('maxDepth', e.target.value)}
                />
            </Field>
            <div className="grid gap-2 grid-cols-2">
                <Field label="Min Samples Split" info={MIN_SAMPLES_SPLIT_INFO}>
                    <Input
                        disabled={disabled}
                        placeholder="Min Samples Split"
                        step={1}
                        min={2}
                        type="number"
                        data-testid="min-samples-split-input"
                        value={minSamplesSplit}
                        onChange={(e) => handleInputChange('minSamplesSplit', e.target.value)}
                    />
                </Field>
                <Field label="Min Samples Leaf" info={MIN_SAMPLES_LEAF_INFO}>
                    <Input
                        disabled={disabled}
                        placeholder="Min Samples Leaf"
                        step={1}
                        min={1}
                        type="number"
                        data-testid="min-samples-leaf-input"
                        value={minSamplesLeaf}
                        onChange={(e) => handleInputChange('minSamplesLeaf', e.target.value)}
                    />
                </Field>
            </div>
            {needsEstimators && (
                <>
                    <Field label="Estimators" info={ESTIMATORS_INFO}>
                        <Input
                            disabled={disabled}
                            placeholder="Estimators"
                            step={1}
                            min={2}
                            type="number"
                            data-testid="estimators-input"
                            value={estimators}
                            onChange={(e) => handleInputChange('estimators', e.target.value)}
                        />
                    </Field>
                </>
            )}
            <div className="grid gap-2 grid-cols-2">
                {needsMaxFeatures && (
                    <Field label="Max Features" info={MAX_FEATURES_INFO}>
                        <Input
                            disabled={disabled}
                            placeholder="Max Features"
                            step={1}
                            min={1}
                            type="number"
                            data-testid="max-features-input"
                            value={maxFeatures}
                            onChange={(e) => handleInputChange('maxFeatures', e.target.value)}
                        />
                    </Field>
                )}
                {needsRandomThresholds && (
                    <Field label="Random Thresholds" info={NUM_RANDOM_THRESHOLDS_INFO}>
                        <Input
                            disabled={disabled}
                            placeholder="Random Thresholds"
                            step={1}
                            min={1}
                            type="number"
                            data-testid="random-thresholds-input"
                            value={numRandomThresholds}
                            onChange={(e) =>
                                handleInputChange('numRandomThresholds', e.target.value)
                            }
                        />
                    </Field>
                )}
            </div>
        </>
    );
}
