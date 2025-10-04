import type { LinearSettings as LinearSettingsType, TaskType } from '@/app/store';
import { LossFunction, Optimizer, Regularization, ThetaInitialization } from '../components';

type LinearSettingsProps = {
    taskType: TaskType;
    settings: LinearSettingsType;
    disabled?: boolean;
    onChange: (config: LinearSettingsType) => void;
};

export default function LinearSettings({
    taskType,
    settings,
    disabled,
    onChange,
}: LinearSettingsProps) {
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
