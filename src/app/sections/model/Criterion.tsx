import { Field } from '@/app/components/ui/field';
import { Input } from '@/app/components/ui/input';
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from '@/app/components/ui/select';
import type { CriterionFunction, TaskType, CriterionFunctionConfig } from '@/app/store';
import type { OptionList } from '../types';

type CriterionProps = {
    taskType: TaskType;
    criterion: CriterionFunctionConfig;
    disabled?: boolean;
    onChange: (config: CriterionFunctionConfig) => void;
};

const DEFAULT_HUBER_DELTA = 1;

const DEFAULT_LINEAR_CRITERION_FUNCTIONS: OptionList = [
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

const DEFAULT_LOGISTIC_CRITERION_FUNCTIONS: OptionList = [
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
                    <SelectTrigger className="w-full truncate">
                        <SelectValue placeholder="Select loss function" />
                    </SelectTrigger>
                    <SelectContent>
                        {criterionFunctions.map((func) => (
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
