import { Button } from '@/app/components/ui/button';
import { Input } from '@/app/components/ui/input';
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from '@/app/components/ui/select';
import { calculateOutputFeatures } from '@/ml/data-processing/transformation';
import { useState, type ChangeEvent } from 'react';

type Transformation = {
    type: string;
    degree: number;
};

type TransformationsProps = {
    transformations: Transformation[];
    numFeatures: number;
    disabled?: boolean;
    onChange: (transformations: Transformation[]) => void;
};

export default function Transformations({
    transformations,
    onChange,
    numFeatures,
    disabled,
}: TransformationsProps) {
    const [localTransformations, setLocalTransformations] = useState(transformations);

    const handleNewTransformation = () => {
        const updatedTransformations = [...localTransformations, { type: '', degree: 1 }];
        setLocalTransformations(updatedTransformations);
    };

    const handleRemoveTransformation = (index: number) => {
        const updatedTransformations = localTransformations.filter((_, i) => i !== index);
        setLocalTransformations(updatedTransformations);
        onChange(updatedTransformations);
    };

    const handleUpdateDegree = (index: number, e: ChangeEvent<HTMLInputElement>) => {
        const updatedTransformations = [...localTransformations];
        updatedTransformations[index].degree = parseInt(e.target.value) || 0;
        setLocalTransformations(updatedTransformations);
        onChange(updatedTransformations);
    };

    const handleUpdateType = (index: number, value: string) => {
        const updatedTransformations = [...localTransformations];
        updatedTransformations[index].type = value;
        setLocalTransformations(updatedTransformations);
        onChange(updatedTransformations);
    };

    return (
        <>
            {localTransformations.map((transformation, index) => {
                const outputFeatures = calculateOutputFeatures(
                    transformation.type,
                    transformation.degree,
                    numFeatures,
                );
                const isPolynomialWithOne =
                    transformation.type === 'polynomial' && transformation.degree === 1;
                return (
                    <div
                        key={index}
                        className="flex flex-col gap-2 rounded-lg border bg-accent/40 p-2"
                    >
                        <div className="grid grid-cols-[2fr_1fr_1fr] gap-2 items-center">
                            <Select
                                disabled={disabled}
                                value={transformation.type}
                                onValueChange={(value) => handleUpdateType(index, value)}
                            >
                                <SelectTrigger className="w-full bg-white">
                                    <SelectValue placeholder="Transform" />
                                </SelectTrigger>
                                <SelectContent>
                                    <SelectItem value="sinusoid">Sinusoid</SelectItem>
                                    <SelectItem value="polynomial">Polynomial</SelectItem>
                                </SelectContent>
                            </Select>
                            <Input
                                className="bg-white"
                                type="number"
                                min={1}
                                placeholder="Degree"
                                disabled={disabled}
                                value={transformation.degree}
                                onChange={(e) => handleUpdateDegree(index, e)}
                            />
                            <Button
                                size="sm"
                                className="px-2 py-1"
                                variant="destructive"
                                disabled={disabled}
                                onClick={() => handleRemoveTransformation(index)}
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
                                With degree 1, the transformation simply returns the original
                                features without any combinations or higher-order terms.
                            </div>
                        )}
                    </div>
                );
            })}

            <Button size="sm" disabled={disabled} onClick={handleNewTransformation}>
                + Add Transformation
            </Button>
        </>
    );
}
