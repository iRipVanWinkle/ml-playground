import { useEffect } from 'react';
import type { ClassificationType as ClassificationTypeName } from '../types';
import { Field, Label, RadioGroup, TooltipWrapper } from '@/app/shared/ui';

type ClassificationTypeProps = {
    classificationType: ClassificationTypeName;
    disabled?: boolean;
    isMulticlass?: boolean;
    onChange: (config: ClassificationTypeName) => void;
};

const CLASSIFICATION_TYPE_INFO = 'Determines how the model handles different numbers of classes.';

const DEFAULT_CLASSIFICATION_TYPES = [
    {
        value: 'binary',
        label: 'Binary Classification (Sigmoid)',
        onlyBinary: true,
        info: 'Predicts one of two classes (yes/no).',
    },
    {
        value: 'softmax',
        label: 'Multiclass Classification (Softmax)',
        info: 'Uses one model to predict which class each example belongs to. Good for problems with multiple classes.',
    },
    {
        value: 'ovr',
        label: 'Multiclass Classification (One-vs-Rest)',
        info: 'Trains one yes/no model for each class. Can work well when classes are very different from each other.',
    },
];

export default function ClassificationType({
    classificationType,
    disabled,
    isMulticlass,
    onChange,
}: ClassificationTypeProps) {
    useEffect(() => {
        const currentClassificationType = DEFAULT_CLASSIFICATION_TYPES.find(
            (type) => type.value === classificationType,
        );
        if (isMulticlass && currentClassificationType?.onlyBinary) {
            onChange('softmax');
        }
    }, [isMulticlass, classificationType, onChange]);

    return (
        <Field label="Classification Type" info={CLASSIFICATION_TYPE_INFO}>
            <RadioGroup
                value={classificationType}
                onValueChange={(value) => onChange(value as ClassificationTypeName)}
                disabled={disabled}
                className="w-full justify-between gap-3 rounded-lg border p-3 transition-colors"
            >
                {DEFAULT_CLASSIFICATION_TYPES.map((model) => {
                    const disabledBinary = model.onlyBinary && isMulticlass;
                    const tooltip = disabledBinary
                        ? 'Binary classification only works with two classes. Use multiclass methods for problems with more classes.'
                        : undefined;

                    return (
                        <TooltipWrapper key={model.value} tooltip={tooltip}>
                            <div className="flex items-center space-x-2">
                                <RadioGroup.Item
                                    value={model.value}
                                    id={model.value}
                                    data-testid={`classification-type-${model.value}`}
                                    disabled={disabledBinary}
                                />
                                <Label
                                    className="font-normal"
                                    htmlFor={model.value}
                                    title={model.info}
                                >
                                    {model.label}
                                </Label>
                            </div>
                        </TooltipWrapper>
                    );
                })}
            </RadioGroup>
        </Field>
    );
}
