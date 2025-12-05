import type { TreeSettings as TreeSettingsType, TreeModelVariant } from '../types';
import { Field, Input, Label, RadioGroup } from '@/app/shared/ui';
import Criterion from './Criterion';
import type { ModelSettingsComponentProps } from '@/app/shared/registry';

const DEFAULT_MODEL_VARIANTS = [
    {
        value: 'decision',
        label: 'Single Decision Tree',
    },
    {
        value: 'bagging',
        label: 'Bagging',
    },
    {
        value: 'forest',
        label: 'Random Forest',
    },
    {
        value: 'extra',
        label: 'Extra Trees',
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
            <Field label="Model Variant">
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
                                <Label className="font-normal" htmlFor={model.value}>
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
            <Field label="Max Depth">
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
                <Field label="Min Samples Split">
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
                <Field label="Min Samples Leaf">
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
                    <Field label="Estimators">
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
                    <Field label="Max Features">
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
                    <Field label="Random Thresholds">
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
