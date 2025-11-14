import type { LogisticSettings as LogisticSettingsType } from '../types';
import {
    LossFunction,
    Optimizer,
    Regularization,
    ThetaInitialization,
} from '@/app/shared/model-settings';
import type { ModelSettingsComponentProps } from '@/app/shared/registry';
import ClassificationType from './ClassificationType';

export function LogisticSettings({
    taskType,
    settings,
    disabled,
    onChange,
    additionalParams,
}: ModelSettingsComponentProps<LogisticSettingsType>) {
    const handleChange = (newSettings: Partial<LogisticSettingsType>) => {
        onChange({ ...newSettings });
    };

    const numCategories = additionalParams?.numCategories ?? 0;

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
