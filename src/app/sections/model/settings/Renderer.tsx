import {
    updateModelSettings,
    useIsTraining,
    useModelSettings,
    useNumCategories,
    useTaskType,
} from '@/app/store';
import LinearSettings from './LinearSettings';
import LogisticSettings from './LogisticSettings';
import NeuralSettings from './NeuralSettings';
import TreeSettings from './TreeSettings';

export default function Renderer() {
    const data = useModelSettings();
    const taskType = useTaskType();
    const isTraining = useIsTraining();
    const numCategories = useNumCategories() ?? 0;

    switch (data.type) {
        case 'linear':
            return (
                <LinearSettings
                    taskType={taskType}
                    settings={data}
                    disabled={isTraining}
                    onChange={(settings) => updateModelSettings(settings)}
                />
            );
        case 'logistic':
            return (
                <LogisticSettings
                    taskType={taskType}
                    numCategories={numCategories}
                    settings={data}
                    disabled={isTraining}
                    onChange={(settings) => updateModelSettings(settings)}
                />
            );
        case 'neural':
            return (
                <NeuralSettings
                    taskType={taskType}
                    settings={data}
                    disabled={isTraining}
                    onChange={(settings) => updateModelSettings(settings)}
                />
            );
        case 'tree':
            return (
                <TreeSettings
                    taskType={taskType}
                    settings={data}
                    disabled={isTraining}
                    onChange={(settings) => updateModelSettings(settings)}
                />
            );
        default:
            return null;
    }
}
