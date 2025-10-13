import { type ModelSettings } from '../store';
import type { TaskType } from '@/app/shared/types';
import LinearSettings from './LinearSettings';
import LogisticSettings from './LogisticSettings';
import NeuralSettings from './NeuralSettings';
import TreeSettings from './TreeSettings';

type RendererProps = {
    taskType: TaskType;
    value: ModelSettings;
    disabled: boolean;
    numCategories: number;
    onChange: (settings: ModelSettings) => void;
};

export default function Renderer({
    taskType,
    value,
    disabled,
    numCategories,
    onChange,
}: RendererProps) {
    switch (value.type) {
        case 'linear':
            return (
                <LinearSettings
                    taskType={taskType}
                    settings={value}
                    disabled={disabled}
                    onChange={onChange}
                />
            );
        case 'logistic':
            return (
                <LogisticSettings
                    taskType={taskType}
                    numCategories={numCategories}
                    settings={value}
                    disabled={disabled}
                    onChange={onChange}
                />
            );
        case 'neural':
            return (
                <NeuralSettings
                    taskType={taskType}
                    settings={value}
                    disabled={disabled}
                    onChange={onChange}
                />
            );
        case 'tree':
            return (
                <TreeSettings
                    taskType={taskType}
                    settings={value}
                    disabled={disabled}
                    onChange={onChange}
                />
            );
        default:
            return null;
    }
}
