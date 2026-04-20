import { Button, Field } from '@/app/shared/ui';
import { setTransformations, useTransformations } from '@/app/store';
import {
    createEmptyTransformation,
    updateTransformationType,
    updateTransformationDegree,
    removeTransformation,
} from '../libs';
import { TransformationItem } from './TransformationItem';
import type { TransformationType } from '@/app/shared/types';

type TransformationBuilderProps = {
    disabled?: boolean;
    numFeatures: number;
};

export function TransformationBuilder({ disabled, numFeatures }: TransformationBuilderProps) {
    const transformations = useTransformations();

    const handleNewTransformation = () => {
        const updatedTransformations = [...transformations, createEmptyTransformation()];
        setTransformations(updatedTransformations);
    };

    const handleRemoveTransformation = (index: number) => {
        const updatedTransformations = removeTransformation(transformations, index);
        setTransformations(updatedTransformations);
    };

    const handleUpdateDegree = (index: number, degree: number) => {
        const updatedTransformations = updateTransformationDegree(transformations, index, degree);
        setTransformations(updatedTransformations);
    };

    const handleUpdateType = (index: number, type: TransformationType) => {
        const updatedTransformations = updateTransformationType(transformations, index, type);
        setTransformations(updatedTransformations);
    };

    return (
        <Field label="Transformations">
            {transformations.map((transformation, index) => (
                <TransformationItem
                    key={index}
                    transformation={transformation}
                    index={index}
                    numFeatures={numFeatures}
                    disabled={disabled}
                    onUpdateType={handleUpdateType}
                    onUpdateDegree={handleUpdateDegree}
                    onRemove={handleRemoveTransformation}
                />
            ))}

            <Button
                size="sm"
                disabled={disabled}
                onClick={handleNewTransformation}
                data-testid="add-transformation-button"
            >
                + Add Transformation
            </Button>
        </Field>
    );
}
