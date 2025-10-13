import type { NeuralSettings as NeuralSettingsType } from '../store';
import type { TaskType } from '@/app/shared/types';
import {
    Layers,
    LossFunction,
    Optimizer,
    Regularization,
    ThetaInitialization,
} from '../components';

type NeuralSettingsProps = {
    taskType: TaskType;
    settings: NeuralSettingsType;
    disabled?: boolean;
    onChange: (config: NeuralSettingsType) => void;
};

export default function NeuralSettings({
    taskType,
    settings,
    disabled,
    onChange,
}: NeuralSettingsProps) {
    const handleChange = (newSettings: Partial<NeuralSettingsType>) => {
        onChange({ ...settings, ...newSettings });
    };

    return (
        <>
            <Layers
                layers={settings.layers}
                disabled={disabled}
                onChange={(layers) => handleChange({ layers })}
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
