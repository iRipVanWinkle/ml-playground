import { useState } from 'react';
import { Button, Field } from '@/app/shared/ui';
import { updateTransformations } from '../store/actions';
import { useTransformations, type TransformationType } from '../store';
import {
    createEmptyTransformation,
    updateTransformationType,
    updateTransformationDegree,
    removeTransformation,
} from '../libs';
import { TransformationItem } from './TransformationItem';

export type TransformationBuilderProps = {
    numFeatures: number;
    disabled?: boolean;
};

export function TransformationBuilder({ numFeatures, disabled }: TransformationBuilderProps) {
    const transformations = useTransformations();
    const [localTransformations, setLocalTransformations] = useState(transformations);

    const handleNewTransformation = () => {
        const updatedTransformations = [...localTransformations, createEmptyTransformation()];
        setLocalTransformations(updatedTransformations);
    };

    const handleRemoveTransformation = (index: number) => {
        const updatedTransformations = removeTransformation(localTransformations, index);
        setLocalTransformations(updatedTransformations);
        updateTransformations(updatedTransformations);
    };

    const handleUpdateDegree = (index: number, degree: number) => {
        const updatedTransformations = updateTransformationDegree(
            localTransformations,
            index,
            degree,
        );
        setLocalTransformations(updatedTransformations);
        updateTransformations(updatedTransformations);
    };

    const handleUpdateType = (index: number, type: TransformationType) => {
        const updatedTransformations = updateTransformationType(localTransformations, index, type);
        setLocalTransformations(updatedTransformations);
        updateTransformations(updatedTransformations);
    };

    return (
        <Field label="Transformations" htmlFor="transformationBuilder">
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
