import type { TaskType } from '@/app/shared/types';
import { getModelRegistry } from '@/app/models/ui-registry';
import { updateModelSettings, useModelSettings } from '@/app/store';

type RendererProps = {
    taskType: TaskType;
    disabled: boolean;
    numCategories?: number;
};

const modelRegistry = getModelRegistry();

export function SettingsRenderer({ taskType, disabled, numCategories }: RendererProps) {
    const settings = useModelSettings();

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
