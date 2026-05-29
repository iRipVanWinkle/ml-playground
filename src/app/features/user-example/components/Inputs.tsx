import { useState } from 'react';
import { Input, Label } from '@/app/shared/ui';

type InputsProps = {
    features: string[];
    inputs?: number[];
    onChange?: (inputs: number[]) => void;
};

export const Inputs = function Inputs({ features, inputs, onChange }: InputsProps) {
    const [prevInputs, setPrevInputs] = useState<number[] | undefined>(inputs);
    const [localInputs, setLocalInputs] = useState<(number | '')[]>(
        () => inputs ?? Array(features.length).fill(''),
    );

    if (prevInputs !== inputs) {
        setPrevInputs(inputs);
        setLocalInputs(() => inputs ?? Array(features.length).fill(''));
    }

    const handleChangeInput = (index: number, value: string) => {
        const next = [...localInputs];
        next[index] = Number(value);
        setLocalInputs(next);

        if (next.every((n): n is number => Number.isFinite(n))) {
            onChange?.(next);
        }
    };

    return (
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            {features.map((feature, index) => (
                <div key={`${feature}-${index}`} className="space-y-1">
                    <Label
                        htmlFor={`${feature}-${index}`}
                        className="text-xs font-normal text-muted-foreground"
                    >
                        {feature}
                    </Label>
                    <Input
                        id={`${feature}-${index}`}
                        type="number"
                        className="text-sm"
                        value={localInputs[index] ?? ''}
                        onChange={(e) => handleChangeInput(index, e.target.value)}
                    />
                </div>
            ))}
        </div>
    );
};
