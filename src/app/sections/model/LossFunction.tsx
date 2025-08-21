import { Field } from '@/app/components/ui/field';
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from '@/app/components/ui/select';
import type { LossFunction as LossFunctionName, LossFunctionConfig, TaskType } from '@/app/store';
import type { OptionList } from '../types';

type LossFunctionProps = {
    taskType: TaskType;
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

const DEFAULT_LOGISTIC_LOSS_FUNCTIONS: OptionList = [
    {
        value: 'binaryCrossentropy',
        label: 'Binary cross-entropy',
    },
    {
        value: 'logitsBasedBinaryCrossentropy',
        label: 'Binary cross-entropy (with logits)',
    },
    {
        value: 'categoricalCrossentropy',
        label: 'Categorical cross-entropy',
    },
    {
        value: 'logitsBasedCategoricalCrossentropy',
        label: 'Categorical cross-entropy (with logits)',
    },
];

export default function LossFunction({
    taskType,
    lossFunction,
    disabled,
    onChange,
}: LossFunctionProps) {
    const handleFunctionChange = (type: LossFunctionName) => {
        onChange({ type });
    };

    const lossFunctions =
        taskType === 'classification'
            ? DEFAULT_LOGISTIC_LOSS_FUNCTIONS
            : DEFAULT_LINEAR_LOSS_FUNCTIONS;

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
