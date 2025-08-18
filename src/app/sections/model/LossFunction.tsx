import { Field } from '@/app/components/ui/field';
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from '@/app/components/ui/select';
import type { LossFunction as LossFunctionName, LossFunctionConfig } from '@/app/store';
import type { OptionList } from '../types';

type LossFunctionProps = {
    lossFunction: LossFunctionConfig;
    disabled?: boolean;
    onChange: (config: LossFunctionConfig) => void;
};

const DEFAULT_LINEAR_LOSS_FUNCTIONS: OptionList = [
    {
        value: 'mse',
        label: 'MSE (Mean Squared Error)',
    },
    {
        value: 'mae',
        label: 'MAE (Mean Absolute Error)',
    },
];

export default function LossFunction({ lossFunction, disabled, onChange }: LossFunctionProps) {
    const handleFunctionChange = (type: LossFunctionName) => {
        onChange({ type });
    };

    const lossFunctions = DEFAULT_LINEAR_LOSS_FUNCTIONS;

    return (
        <div className="grid gap-2">
            <Field label="Loss Function">
                <Select
                    disabled={disabled}
                    value={lossFunction.type as string}
                    onValueChange={(value) => handleFunctionChange(value as LossFunctionName)}
                >
                    <SelectTrigger className="w-full truncate">
                        <SelectValue placeholder="Select loss function" />
                    </SelectTrigger>
                    <SelectContent>
                        {lossFunctions.map((func) => (
                            <SelectItem
                                key={func.value}
                                value={func.value}
                                disabled={func.disabled}
                            >
                                {func.label}
                            </SelectItem>
                        ))}
                    </SelectContent>
                </Select>
            </Field>
        </div>
    );
}
