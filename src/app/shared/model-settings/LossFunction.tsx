import { Field, Input, Select } from '@/app/shared/ui';
import type { LossFunctionType, LossFunctionConfig } from '@/ml/factories';
import type { TaskType } from '@/app/shared/types';

type LossFunctionProps = {
    taskType: TaskType;
    lossFunction: LossFunctionConfig;
    disabled?: boolean;
    onChange: (config: LossFunctionConfig) => void;
};

const DEFAULT_HUBER_DELTA = 1;

const DEFAULT_LINEAR_LOSS_FUNCTIONS = [
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
    {
        value: 'logcosh',
        label: 'Log-Cosh',
    },
];

const DEFAULT_LOGISTIC_LOSS_FUNCTIONS = [
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
    const handleFunctionChange = (type: LossFunctionType) => {
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
            <Field label="Loss Function" htmlFor="lossFunctionSelect">
                <Select
                    disabled={disabled}
                    value={lossFunction.type as string}
                    onValueChange={(value) => handleFunctionChange(value as LossFunctionType)}
                >
                    <Select.Trigger
                        id="lossFunctionSelect"
                        className="w-full truncate"
                        data-testid="loss-function-select"
                    >
                        <Select.Value placeholder="Select loss function" />
                    </Select.Trigger>
                    <Select.Content>
                        {lossFunctions.map((func) => (
                            <Select.Item key={func.value} value={func.value}>
                                {func.label}
                            </Select.Item>
                        ))}
                    </Select.Content>
                </Select>
            </Field>
            {lossFunction.type === 'huber' && (
                <Field label="Delta" htmlFor="huberDeltaInput">
                    <Input
                        id="huberDeltaInput"
                        data-testid="huber-delta-input"
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
