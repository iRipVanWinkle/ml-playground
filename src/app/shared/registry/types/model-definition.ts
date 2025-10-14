import type { ComponentType } from 'react';
import type { ModelType } from '@/app/models/types';
import type { TaskType } from '@/app/shared/types';
import type { SettingsOf } from './utils';

export interface ModelDefinition<TKey extends ModelType = ModelType> {
    key: TKey;
    label: string;
    taskTypes: TaskType[];

    defaultSettings: (taskType?: TaskType) => SettingsOf<TKey>;
    settingsComponent: ComponentType<ModelSettingsComponentProps<SettingsOf<TKey>>>;
}

export type ModelSettingsComponentProps<TSettings> = {
    taskType: TaskType;
    settings: TSettings;
    disabled: boolean;
    additionalParams?: {
        numCategories?: number;
    };
    onChange: (settings: TSettings) => void;
};
