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

type Layers = { units: number; activation?: string };

type LayersProps = {
    layers: Layers[];
    disabled?: boolean;
    onChange: (layers: Layers[]) => void;
};

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
                        className="flex flex-col gap-2 rounded-lg border bg-accent/40 p-2"
                    >
                        <div className="grid grid-cols-[1fr_2fr_1fr] gap-2 items-center">
                            <Input
                                className="bg-white"
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
                                <SelectTrigger className="w-full bg-white">
                                    <SelectValue placeholder="Activation" />
                                </SelectTrigger>
                                <SelectContent>
                                    <SelectItem value="linear">Linear</SelectItem>
                                    <SelectItem value="relu" title="Rectified Linear Unit">
                                        ReLU
                                    </SelectItem>
                                    <SelectItem value="sigmoid">Sigmoid</SelectItem>
                                    <SelectItem value="tanh" title="Hyperbolic Tangent">
                                        Tanh
                                    </SelectItem>
                                    <SelectItem value="softmax">Softmax</SelectItem>
                                </SelectContent>
                            </Select>

                            <Button
                                size="sm"
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

            <Button size="sm" disabled={disabled} onClick={handleNewTransformation}>
                + Add Layer
            </Button>
        </Field>
    );
}
