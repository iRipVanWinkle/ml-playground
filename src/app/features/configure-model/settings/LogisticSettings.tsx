import type { LogisticSettings as LogisticSettingsType, TaskType } from '@/app/store';
import {
    ClassificationType,
    LossFunction,
    Optimizer,
    Regularization,
    ThetaInitialization,
} from '../components';

type LogisticSettingsProps = {
    taskType: TaskType;
    numCategories: number;
    settings: LogisticSettingsType;
    disabled?: boolean;
    onChange: (config: LogisticSettingsType) => void;
};

export default function LogisticSettings({
    taskType,
    numCategories,
    settings,
    disabled,
    onChange,
}: LogisticSettingsProps) {
    const handleChange = (newSettings: Partial<LogisticSettingsType>) => {
        onChange({ ...settings, ...newSettings });
    };

    return (
        <>
            <ClassificationType
                classificationType={settings.classificationType}
                disabled={disabled}
                isMulticlass={numCategories > 2}
                onChange={(classificationType) => handleChange({ classificationType })}
            />
            <LossFunction
                taskType={taskType}
                lossFunction={settings.lossFunction}
                disabled={disabled}
                onChange={(lossFunction) => handleChange({ lossFunction })}
            />
            <Optimizer
                optimizer={settings.optimizer}
                disabled={disabled}
                onChange={(optimizer) => handleChange({ optimizer })}
            />
            <Regularization
                regularization={settings.regularization}
                disabled={disabled}
                onChange={(regularization) => handleChange({ regularization })}
            />
            <ThetaInitialization
                thetaInitialization={settings.thetaInitialization}
                disabled={disabled}
                onChange={(thetaInitialization) => handleChange({ thetaInitialization })}
            />
        </>
    );
}
