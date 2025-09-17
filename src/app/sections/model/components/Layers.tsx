import { Button } from '@/app/components/ui/button';
import { Field } from '@/app/components/ui/field';
import { Input } from '@/app/components/ui/input';
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from '@/app/components/ui/select';
import { useState, type ChangeEvent } from 'react';
import type { OptionList } from '../../types';

type Layers = { units: number; activation?: string };

type LayersProps = {
    layers: Layers[];
    disabled?: boolean;
    onChange: (layers: Layers[]) => void;
};

const DEFAULT_ACTIVATION_FUNCTIONS: OptionList = [
    { value: 'linear', label: 'Linear' },
    { value: 'relu', label: 'ReLU', title: 'Rectified Linear Unit' },
    { value: 'sigmoid', label: 'Sigmoid' },
    { value: 'tanh', label: 'Tanh', title: 'Hyperbolic Tangent' },
    { value: 'softmax', label: 'Softmax' },
];

export default function Layers({ layers, onChange, disabled }: LayersProps) {
    const [localLayers, setLocalLayers] = useState(layers);

    const handleNewTransformation = () => {
        const previousLayer = localLayers.at(-1);
        const updatedLayers = [
            ...localLayers,
            {
                units: previousLayer?.units ?? 1,
                activation: previousLayer?.activation,
            },
        ];
        setLocalLayers(updatedLayers);
        onChange(updatedLayers);
    };

    const handleRemoveTransformation = (index: number) => {
        const updatedLayers = localLayers.filter((_, i) => i !== index);
        setLocalLayers(updatedLayers);
        onChange(updatedLayers);
    };

    const handleUpdateUnits = (index: number, e: ChangeEvent<HTMLInputElement>) => {
        const updatedLayers = [...localLayers];
        updatedLayers[index].units = parseInt(e.target.value) || 0;
        setLocalLayers(updatedLayers);
        onChange(updatedLayers);
    };

    const handleUpdateActivation = (index: number, value: string) => {
        const updatedLayers = [...localLayers];
        updatedLayers[index].activation = value;
        setLocalLayers(updatedLayers);
        onChange(updatedLayers);
    };

    return (
        <Field label="Layers">
            {localLayers.map((layer, index) => {
                return (
                    <div
                        key={index}
                        data-testid="layer-item"
                        className="flex flex-col gap-2 rounded-lg border bg-accent/40 p-2"
                    >
                        <div className="grid grid-cols-[1fr_2fr_1fr] gap-2 items-center">
                            <Input
                                className="bg-white"
                                data-testid="units-input"
                                type="number"
                                min={1}
                                placeholder="Units"
                                disabled={disabled}
                                value={layer.units}
                                onChange={(e) => handleUpdateUnits(index, e)}
                            />

                            <Select
                                disabled={disabled}
                                value={layer.activation}
                                onValueChange={(value) => handleUpdateActivation(index, value)}
                            >
                                <SelectTrigger
                                    className="w-full bg-white"
                                    data-testid="activation-select"
                                >
                                    <SelectValue placeholder="Activation" />
                                </SelectTrigger>
                                <SelectContent>
                                    {DEFAULT_ACTIVATION_FUNCTIONS.map((item) => (
                                        <SelectItem
                                            key={item.value}
                                            value={item.value}
                                            title={item.title}
                                        >
                                            {item.label}
                                        </SelectItem>
                                    ))}
                                </SelectContent>
                            </Select>

                            <Button
                                size="sm"
                                data-testid="remove-layer-button"
                                className="px-2 py-1"
                                variant="destructive"
                                disabled={disabled || localLayers.length === 1}
                                onClick={() => handleRemoveTransformation(index)}
                            >
                                Remove
                            </Button>
                        </div>
                    </div>
                );
            })}

            <Button
                size="sm"
                disabled={disabled}
                onClick={handleNewTransformation}
                data-testid="add-layer-button"
            >
                + Add Layer
            </Button>
        </Field>
    );
}
