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
import { Input } from '@/app/components/ui/input';

type LossFunctionProps = {
    taskType: TaskType;
    lossFunction: LossFunctionConfig;
    disabled?: boolean;
    onChange: (config: LossFunctionConfig) => void;
};

const DEFAULT_HUBER_DELTA = 1;

const DEFAULT_LINEAR_LOSS_FUNCTIONS: OptionList = [
    {
        value: 'mse',
        label: 'MSE (Mean Squared Error)',
    },
    {
        value: 'mae',
        label: 'MAE (Mean Absolute Error)',
    },
    {
        value: 'huber',
        label: 'Huber',
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
        if (type === 'huber') {
            onChange({ type: 'huber', delta: DEFAULT_HUBER_DELTA });
        } else {
            onChange({ type });
        }
    };

    const lossFunctions =
        taskType === 'classification'
            ? DEFAULT_LOGISTIC_LOSS_FUNCTIONS
            : DEFAULT_LINEAR_LOSS_FUNCTIONS;

    let containerClass = 'grid gap-2';
    if (lossFunction.type === 'huber') {
        containerClass += ' grid-cols-2';
    }

    return (
        <div className={containerClass}>
            <Field label="Loss Function" htmlFor="loss-function">
                <Select
                    disabled={disabled}
                    value={lossFunction.type as string}
                    onValueChange={(value) => handleFunctionChange(value as LossFunctionName)}
                >
                    <SelectTrigger id="loss-function" className="w-full truncate">
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
            {lossFunction.type === 'huber' && (
                <Field label="Delta" htmlFor="huber-delta">
                    <Input
                        id="huber-delta"
                        disabled={disabled}
                        placeholder="Delta"
                        step={0.1}
                        type="number"
                        value={lossFunction.delta}
                        onChange={(e) =>
                            onChange({ ...lossFunction, delta: parseFloat(e.target.value) })
                        }
                    />
                </Field>
            )}
        </div>
    );
}
