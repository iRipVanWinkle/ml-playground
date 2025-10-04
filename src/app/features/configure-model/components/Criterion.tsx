import { Field, Input, Select } from '@/app/shared/ui';
import type { CriterionFunction, TaskType, CriterionFunctionConfig } from '@/app/store';

type CriterionProps = {
    taskType: TaskType;
    criterion: CriterionFunctionConfig;
    disabled?: boolean;
    onChange: (config: CriterionFunctionConfig) => void;
};

const DEFAULT_HUBER_DELTA = 1;

const DEFAULT_LINEAR_CRITERION_FUNCTIONS = [
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

const DEFAULT_LOGISTIC_CRITERION_FUNCTIONS = [
    {
        value: 'gini',
        label: 'Gini',
    },
    {
        value: 'entropy',
        label: 'Entropy',
    },
];

export default function Criterion({ taskType, criterion, disabled, onChange }: CriterionProps) {
    const handleFunctionChange = (type: CriterionFunction) => {
        if (type === 'huber') {
            onChange({ type: 'huber', delta: DEFAULT_HUBER_DELTA });
        } else {
            onChange({ type });
        }
    };

    const criterionFunctions =
        taskType === 'regression'
            ? DEFAULT_LINEAR_CRITERION_FUNCTIONS
            : DEFAULT_LOGISTIC_CRITERION_FUNCTIONS;

    let containerClass = 'grid gap-2';
    if (criterion.type === 'huber') {
        containerClass += ' grid-cols-2';
    }

    return (
        <div className={containerClass}>
            <Field label="Criterion">
                <Select
                    disabled={disabled}
                    value={criterion.type as string}
                    onValueChange={(value) => handleFunctionChange(value as CriterionFunction)}
                >
                    <Select.Trigger className="w-full truncate">
                        <Select.Value placeholder="Select loss function" />
                    </Select.Trigger>
                    <Select.Content>
                        {criterionFunctions.map((func) => (
                            <Select.Item key={func.value} value={func.value}>
                                {func.label}
                            </Select.Item>
                        ))}
                    </Select.Content>
                </Select>
            </Field>
            {criterion.type === 'huber' && (
                <Field label="Delta">
                    <Input
                        disabled={disabled}
                        placeholder="Delta (for Huber)"
                        step={0.1}
                        type="number"
                        value={criterion.delta}
                        onChange={(e) =>
                            onChange({ ...criterion, delta: parseFloat(e.target.value) })
                        }
                    />
                </Field>
            )}
        </div>
    );
}
