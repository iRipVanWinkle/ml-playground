import { startTransition, useEffect, useState } from 'react';
import { Button, Field } from '@/app/shared/ui';
import { useTransformations } from '../store';
import { resetTransformations, updateTransformations } from '../store/actions';
import {
    createEmptyTransformation,
    updateTransformationType,
    updateTransformationDegree,
    removeTransformation,
} from '../libs';
import { TransformationItem } from './TransformationItem';
import type { TaskType, TransformationType } from '@/app/shared/types';

type TransformationBuilderProps = {
    disabled?: boolean;
    numFeatures: number;
    taskType: TaskType;
};

export function TransformationBuilder({
    disabled,
    numFeatures,
    taskType,
}: TransformationBuilderProps) {
    const transformations = useTransformations();
    const [localTransformations, setLocalTransformations] = useState(transformations);

    useEffect(() => {
        startTransition(() => {
            resetTransformations();
            setLocalTransformations([]);
        });
    }, [taskType]);

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
