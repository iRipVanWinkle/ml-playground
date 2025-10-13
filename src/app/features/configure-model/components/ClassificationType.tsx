import { useEffect } from 'react';
import type { ClassificationType as ClassificationTypeName } from '../store';
import { Field, Label, RadioGroup, TooltipWrapper } from '@/app/shared/ui';

type ClassificationTypeProps = {
    classificationType: ClassificationTypeName;
    disabled?: boolean;
    isMulticlass?: boolean;
    onChange: (config: ClassificationTypeName) => void;
};

const DEFAULT_CLASSIFICATION_TYPES = [
    {
        value: 'binary',
        label: 'Binary Classification (Sigmoid)',
        onlyBinary: true,
    },
    {
        value: 'softmax',
        label: 'Multiclass Classification (Softmax)',
    },
    {
        value: 'ovr',
        label: 'Multiclass Classification (One-vs-Rest)',
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
        <Field label="Classification Type">
            <RadioGroup
                value={classificationType}
                onValueChange={(value) => onChange(value as ClassificationTypeName)}
                disabled={disabled}
                className="w-full justify-between gap-3 rounded-lg border p-3 transition-colors"
            >
                {DEFAULT_CLASSIFICATION_TYPES.map((model) => {
                    const disabledBinary = model.onlyBinary && isMulticlass;
                    const tooltip = disabledBinary
                        ? 'This option is not suitable for multiclass classification.'
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
                                <Label className="font-normal" htmlFor={model.value}>
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
