import { useState } from 'react';
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
    const [localTransformations, setLocalTransformations] = useState(transformations);

    const handleNewTransformation = () => {
        const updatedTransformations = [...localTransformations, createEmptyTransformation()];
        setLocalTransformations(updatedTransformations);
    };

    const handleRemoveTransformation = (index: number) => {
        const updatedTransformations = removeTransformation(localTransformations, index);
        setLocalTransformations(updatedTransformations);
        setTransformations(updatedTransformations);
    };

    const handleUpdateDegree = (index: number, degree: number) => {
        const updatedTransformations = updateTransformationDegree(
            localTransformations,
            index,
            degree,
        );
        setLocalTransformations(updatedTransformations);
        setTransformations(updatedTransformations);
    };

    const handleUpdateType = (index: number, type: TransformationType) => {
        const updatedTransformations = updateTransformationType(localTransformations, index, type);
        setLocalTransformations(updatedTransformations);
        setTransformations(updatedTransformations);
    };

    return (
        <Field label="Transformations">
            {localTransformations.map((transformation, index) => (
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
