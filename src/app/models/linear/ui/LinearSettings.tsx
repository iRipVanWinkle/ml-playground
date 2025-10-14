import type { ModelSettingsComponentProps } from '@/app/shared/registry/types/model-definition';
import type { LinearSettings as LinearSettingsType } from '../types';
import {
    LossFunction,
    Optimizer,
    Regularization,
    ThetaInitialization,
} from '@/app/shared/model-settings';

export function LinearSettings({
    taskType,
    settings,
    disabled,
    onChange,
}: ModelSettingsComponentProps<LinearSettingsType>) {
    const handleChange = (newSettings: Partial<LinearSettingsType>) => {
        onChange({ ...settings, ...newSettings });
    };

    return (
        <>
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
