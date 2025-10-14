import type { TaskType } from '@/app/shared/types';
import { getModelRegistry } from '@/app/models/ui-registry';
import { useModelSettingsStore } from '../store';
import { updateModelSettings } from '../store/actions';

type RendererProps = {
    taskType: TaskType;
    disabled: boolean;
    numCategories?: number;
};

const modelRegistry = getModelRegistry();

export function SettingsRenderer({ taskType, disabled, numCategories }: RendererProps) {
    const settings = useModelSettingsStore();

    const modelDefinition = modelRegistry.get(settings.type);
    const SettingsComponent = modelDefinition.settingsComponent;

    return (
        <SettingsComponent
            taskType={taskType}
            settings={settings}
            disabled={disabled}
            additionalParams={{ numCategories }}
            onChange={(settings) => updateModelSettings(settings)}
        />
    );
}
