import { Card, CardContent, CardHeader, CardTitle } from '@/app/components/ui/card';
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from '@/app/components/ui/select';
import {
    setModelType,
    updateModelSettings,
    useIsTraining,
    useModelSettings,
    useTaskType,
    type ModelType,
} from '@/app/store';
import { Field } from '@/app/components/ui/field';
import Optimizer from './model/Optimizer';
import LossFunction from './model/LossFunction';
import Regularization from './model/Regularization';

type OptionList = Array<{
    value: string;
    label: string;
    disabled?: boolean;
}>;

const DEFAULT_REGRESSION_MODEL_TYPES = [
    {
        value: 'linear',
        label: 'Linear Regression',
    },
] as OptionList;

const DEFAULT_CLASSIFICATION_MODEL_TYPES = [
    {
        value: 'logistic',
        label: 'Logistic Regression',
    },
] as OptionList;

export default function ModelSection() {
    const data = useModelSettings();
    const taskType = useTaskType();
    const isTraining = useIsTraining();
    const handleChange = updateModelSettings;

    const modelTypes =
        taskType === 'regression'
            ? DEFAULT_REGRESSION_MODEL_TYPES
            : DEFAULT_CLASSIFICATION_MODEL_TYPES;

    return (
        <Card className="gap-5">
            <CardHeader>
                <CardTitle>Model</CardTitle>
            </CardHeader>
            <CardContent className="grid gap-5">
                <Field label="Model Type">
                    <Select
                        disabled={isTraining}
                        value={data.type}
                        onValueChange={(value) => setModelType(value as ModelType)}
                    >
                        <SelectTrigger className="w-full truncate">
                            <SelectValue placeholder="Select Model Type" />
                        </SelectTrigger>
                        <SelectContent>
                            {modelTypes.map((model) => (
                                <SelectItem
                                    key={model.value}
                                    value={model.value}
                                    disabled={model.disabled}
                                >
                                    {model.label}
                                </SelectItem>
                            ))}
                        </SelectContent>
                    </Select>
                </Field>

                <LossFunction
                    taskType={taskType}
                    lossFunction={data.lossFunction}
                    disabled={isTraining}
                    onChange={(lossFunction) => handleChange({ lossFunction })}
                />

                <Optimizer
                    optimizer={data.optimizer}
                    disabled={isTraining}
                    onChange={(optimizer) => handleChange({ optimizer })}
                />

                <Regularization
                    regularization={data.regularization}
                    disabled={isTraining}
                    onChange={(regularization) => handleChange({ regularization })}
                />
            </CardContent>
        </Card>
    );
}
