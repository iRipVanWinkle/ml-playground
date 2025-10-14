import type { NeuralSettings as NeuralSettingsType } from '../types';
import {
    LossFunction,
    Optimizer,
    Regularization,
    ThetaInitialization,
} from '@/app/shared/model-settings';
import type { ModelSettingsComponentProps } from '@/app/shared/registry';
import Layers from './Layers';

export function NeuralSettings({
    taskType,
    settings,
    disabled,
    onChange,
}: ModelSettingsComponentProps<NeuralSettingsType>) {
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
