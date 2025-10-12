import { type ChangeEvent } from 'react';
import { Button, Input, Select } from '@/app/shared/ui';
import { TRANSFORMATION_TYPES } from '../constants';
import { calculateTransformationOutputFeatures, isPolynomialWithDegreeOne } from '../libs';
import type { Transformation, TransformationType } from '../store';

export type TransformationItemProps = {
    transformation: Transformation;
    index: number;
    numFeatures: number;
    disabled?: boolean;
    onUpdateType: (index: number, type: TransformationType) => void;
    onUpdateDegree: (index: number, degree: number) => void;
    onRemove: (index: number) => void;
};

export function TransformationItem({
    transformation,
    index,
    numFeatures,
    disabled,
    onUpdateType,
    onUpdateDegree,
    onRemove,
}: TransformationItemProps) {
    const outputFeatures = calculateTransformationOutputFeatures(
        transformation.type,
        transformation.degree,
        numFeatures,
    );
    const isPolynomialWithOne = isPolynomialWithDegreeOne(
        transformation.type,
        transformation.degree,
    );

    const handleDegreeChange = (e: ChangeEvent<HTMLInputElement>) => {
        const degree = parseInt(e.target.value) || 0;
        onUpdateDegree(index, degree);
    };

    const handleTypeChange = (value: string) => {
        onUpdateType(index, value as TransformationType);
    };

    return (
        <div
            data-testid="transformation-container"
            className="flex flex-col gap-2 rounded-lg border bg-accent/40 p-2"
        >
            <div className="grid grid-cols-[2fr_1fr_1fr] gap-2 items-center">
                <Select
                    disabled={disabled}
                    value={transformation.type}
                    onValueChange={handleTypeChange}
                >
                    <Select.Trigger
                        className="w-full bg-white"
                        data-testid="transformation-type-select"
                    >
                        <Select.Value placeholder="Transform" />
                    </Select.Trigger>
                    <Select.Content>
                        {TRANSFORMATION_TYPES.map((transformation) => (
                            <Select.Item key={transformation.value} value={transformation.value}>
                                {transformation.label}
                            </Select.Item>
                        ))}
                    </Select.Content>
                </Select>
                <Input
                    data-testid="degree-input"
                    className="bg-white"
                    type="number"
                    min={1}
                    placeholder="Degree"
                    disabled={disabled}
                    value={transformation.degree}
                    onChange={handleDegreeChange}
                />
                <Button
                    size="sm"
                    data-testid="remove-transformation-button"
                    aria-label={`Remove transformation ${index + 1}`}
                    className="px-2 py-1"
                    variant="destructive"
                    disabled={disabled}
                    onClick={() => onRemove(index)}
                >
                    Remove
                </Button>
            </div>
            {outputFeatures > 0 && (
                <div className="text-xs text-left text-muted-foreground">
                    Output features: <b>{outputFeatures}</b>
                </div>
            )}
            {isPolynomialWithOne && (
                <div className="text-xs text-left text-amber-600">
                    With degree 1, the transformation simply returns the original features without
                    any combinations or higher-order terms.
                </div>
            )}
        </div>
    );
}
