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
const LOSS_FUNCTION_INFO =
    'Measures how far predictions are from true values. Training tries to make this as small as possible.';

const DEFAULT_LINEAR_LOSS_FUNCTIONS = [
    {
        value: 'mse' as const,
        label: 'MSE (Mean Squared Error)',
        info: 'Punishes large errors more by squaring them.',
    },
    {
        value: 'mae' as const,
        label: 'MAE (Mean Absolute Error)',
        info: 'Measures average size of errors equally.',
    },
    {
        value: 'huber' as const,
        label: 'Huber',
        info: 'Combines MSE and MAE. Uses MSE for small errors and MAE for large errors.',
    },
    {
        value: 'logcosh' as const,
        label: 'Log-Cosh',
        info: 'Similar to MAE but smoother. Less affected by unusual data points that are very different from the rest.',
    },
];

const DEFAULT_LOGISTIC_LOSS_FUNCTIONS = [
    {
        value: 'binaryCrossentropy',
        label: 'Binary cross-entropy',
        info: 'For yes/no problems with probability inputs (0 to 1).',
    },
    {
        value: 'logitsBasedBinaryCrossentropy',
        label: 'Binary cross-entropy (with logits)',
        info: 'For yes/no problems with raw scores (applies sigmoid internally).',
    },
    {
        value: 'categoricalCrossentropy',
        label: 'Categorical cross-entropy',
        info: 'For multiple classes with probability inputs (sum to 1).',
    },
    {
        value: 'logitsBasedCategoricalCrossentropy',
        label: 'Categorical cross-entropy (with logits)',
        info: 'For multiple classes with raw scores (applies softmax internally).',
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
            <Field label="Loss Function" htmlFor="lossFunctionSelect" info={LOSS_FUNCTION_INFO}>
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
                            <Select.Item key={func.value} value={func.value} title={func.info}>
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
