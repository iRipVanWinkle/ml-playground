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
    useIsTraining,
    useModelSettings,
    useTaskType,
    type ModelType,
} from '@/app/store';
import { Field } from '@/app/components/ui/field';
import { SettingsRenderer } from './settings';

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
    {
        value: 'neural',
        label: 'Neural Networks',
    },
    {
        value: 'tree',
        label: 'Decision Tree',
    },
] as OptionList;

const DEFAULT_CLASSIFICATION_MODEL_TYPES = [
    {
        value: 'logistic',
        label: 'Logistic Regression',
    },
    {
        value: 'neural',
        label: 'Neural Networks',
    },
    {
        value: 'tree',
        label: 'Decision Tree',
    },
] as OptionList;

export default function ModelSection() {
    const data = useModelSettings();
    const taskType = useTaskType();
    const isTraining = useIsTraining();

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
                <Field label="Model Type" htmlFor="modelType">
                    <Select
                        disabled={isTraining}
                        value={data.type}
                        onValueChange={(value) => setModelType(value as ModelType)}
                    >
                        <SelectTrigger
                            className="w-full truncate"
                            id="modelType"
                            data-testid="model-type-select"
                        >
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

                <SettingsRenderer />
            </CardContent>
        </Card>
    );
}
