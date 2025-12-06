import { Field, Input, Select } from '@/app/shared/ui';
import type { CriterionType, CriterionConfig } from '@/ml/factories';
import type { TaskType } from '@/app/shared/types';

type CriterionProps = {
    taskType: TaskType;
    criterion: CriterionConfig;
    disabled?: boolean;
    onChange: (config: CriterionConfig) => void;
};

const DEFAULT_HUBER_DELTA = 1;
const CRITERION_INFO = 'Determines how the tree decides the best way to split data.';

const DEFAULT_LINEAR_CRITERION_FUNCTIONS = [
    {
        value: 'mse',
        label: 'MSE (Mean Squared Error)',
        info: 'Splits to minimize squared errors. Can be affected by outliers.',
    },
    {
        value: 'mae',
        label: 'MAE (Mean Absolute Error)',
        info: 'Splits to minimize absolute errors. Handles unusual data points well without being affected too much.',
    },
    {
        value: 'huber',
        label: 'Huber',
        info: 'Combines MSE and MAE for a balanced approach to handling errors.',
    },
    {
        value: 'logcosh',
        label: 'Log-Cosh',
        info: 'Smooth way to measure errors. More robust to outliers than MSE.',
    },
];

const DEFAULT_LOGISTIC_CRITERION_FUNCTIONS = [
    {
        value: 'gini',
        label: 'Gini',
        info: 'Measures how mixed the classes are in each group. Faster to compute than entropy.',
    },
    {
        value: 'entropy',
        label: 'Entropy',
        info: 'Measures how much information is gained by splitting. Tends to create trees with balanced splits.',
    },
];

export default function Criterion({ taskType, criterion, disabled, onChange }: CriterionProps) {
    const handleFunctionChange = (type: CriterionType) => {
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
            <Field label="Criterion" info={CRITERION_INFO}>
                <Select
                    disabled={disabled}
                    value={criterion.type as string}
                    onValueChange={(value) => handleFunctionChange(value as CriterionType)}
                >
                    <Select.Trigger className="w-full truncate" data-testid="criterion-select">
                        <Select.Value placeholder="Select loss function" />
                    </Select.Trigger>
                    <Select.Content>
                        {criterionFunctions.map((func) => (
                            <Select.Item key={func.value} value={func.value} title={func.info}>
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
                        data-testid="huber-delta-input"
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
