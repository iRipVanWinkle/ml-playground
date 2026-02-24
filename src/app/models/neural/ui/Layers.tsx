import { Button, Field, Input, Select } from '@/app/shared/ui';
import { useState, type ChangeEvent } from 'react';

type Layers = { units: number; activation?: string };

type LayersProps = {
    layers: Layers[];
    disabled?: boolean;
    onChange: (layers: Layers[]) => void;
};

const LAYERS_INFO = 'Define the neural network structure: neurons and activation per layer.';

const DEFAULT_ACTIVATION_FUNCTIONS = [
    {
        value: 'linear',
        label: 'Linear',
        info: 'No change to the input. Good for predicting numbers (regression problems).',
    },
    {
        value: 'relu',
        label: 'ReLU',
        title: 'Rectified Linear Unit',
        info: 'Outputs the input if positive, otherwise outputs zero. Most common for hidden layers in the network.',
    },
    {
        value: 'sigmoid',
        label: 'Sigmoid',
        info: 'Converts output to a value between 0 and 1. Good for yes/no problems (binary classification).',
    },
    {
        value: 'tanh',
        label: 'Tanh',
        title: 'Hyperbolic Tangent',
        info: 'Transforms output to a range between -1 and 1. Similar to sigmoid but centered at zero instead of 0.5.',
    },
    {
        value: 'softmax',
        label: 'Softmax',
        info: 'Outputs probabilities for each class. Good for problems with multiple classes.',
    },
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
        <Field label="Layers" info={LAYERS_INFO}>
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
                                aria-label={`Layer ${index + 1} units`}
                            />

                            <Select
                                disabled={disabled}
                                value={layer.activation}
                                onValueChange={(value) => handleUpdateActivation(index, value)}
                            >
                                <Select.Trigger
                                    className="w-full bg-white"
                                    data-testid="activation-select"
                                    aria-label={`Layer ${index + 1} activation function`}
                                >
                                    <Select.Value placeholder="Activation" />
                                </Select.Trigger>
                                <Select.Content>
                                    {DEFAULT_ACTIVATION_FUNCTIONS.map((item) => (
                                        <Select.Item
                                            key={item.value}
                                            value={item.value}
                                            title={item.info}
                                        >
                                            {item.label}
                                        </Select.Item>
                                    ))}
                                </Select.Content>
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
