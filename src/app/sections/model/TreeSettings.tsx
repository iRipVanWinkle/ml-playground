import { Field } from '@/app/components/ui/field';
import { Input } from '@/app/components/ui/input';
import type {
    LossFunctionConfig,
    TaskType,
    TreeModelVariant,
    TreeSettings,
    TreeSettings as TreeSettingsType,
} from '@/app/store';
import type { OptionList } from '../types';
import { RadioGroup, RadioGroupItem } from '@/app/components/ui/radio-group';
import { Label } from '@/app/components/ui/label';
import Criterion from './Criterion';

type TreeSettingsProps = {
    taskType: TaskType;
    settings: TreeSettingsType;
    lossFunction: LossFunctionConfig;
    disabled?: boolean;
    onChange: (config: TreeSettingsType) => void;
    onChangeLossFunction: (config: LossFunctionConfig) => void;
};

const DEFAULT_MODEL_VARIANTS: OptionList = [
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
] as OptionList;

export default function TreeSettings({
    taskType,
    settings,
    disabled,
    onChange,
}: TreeSettingsProps) {
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

    const handleModelVariantChange = (value: string) => {
        onChange({ ...settings, modelVariant: value as TreeModelVariant });
    };

    // Handle input changes for optimizer parameters
    const handleInputChange = (key: keyof TreeSettings, value: string) => {
        const newConfig = { ...settings, [key]: parseInt(value) };
        onChange(newConfig as TreeSettings);
    };

    return (
        <>
            {/* <div className="grid gap-2"> */}
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
                                <RadioGroupItem
                                    value={model.value}
                                    id={model.value}
                                    disabled={model.disabled}
                                />
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
                        value={minSamplesLeaf}
                        onChange={(e) => handleInputChange('minSamplesLeaf', e.target.value)}
                    />
                </Field>
            </div>
            {modelVariant !== 'decision' && (
                <>
                    <Field label="Estimators">
                        <Input
                            disabled={disabled}
                            placeholder="Estimators"
                            step={1}
                            min={2}
                            type="number"
                            value={estimators}
                            onChange={(e) => handleInputChange('estimators', e.target.value)}
                        />
                    </Field>
                    {(modelVariant === 'forest' || modelVariant === 'extra') && (
                        <div className="grid gap-2 grid-cols-2">
                            <Field label="Max Features">
                                <Input
                                    disabled={disabled}
                                    placeholder="Max Features"
                                    step={1}
                                    min={1}
                                    type="number"
                                    value={maxFeatures}
                                    onChange={(e) =>
                                        handleInputChange('maxFeatures', e.target.value)
                                    }
                                />
                            </Field>
                            {modelVariant === 'extra' && (
                                <Field label="Random Thresholds">
                                    <Input
                                        disabled={disabled}
                                        placeholder="Random Thresholds"
                                        step={1}
                                        min={1}
                                        type="number"
                                        value={numRandomThresholds}
                                        onChange={(e) =>
                                            handleInputChange('numRandomThresholds', e.target.value)
                                        }
                                    />
                                </Field>
                            )}
                        </div>
                    )}
                </>
            )}
        </>
    );
}
