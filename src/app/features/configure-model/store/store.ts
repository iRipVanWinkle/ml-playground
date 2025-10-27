import { create } from 'zustand';
import type { ModelSettings } from '@/app/models/types';
import { linearModelDefinition } from '@/app/models/linear/ui.definition';

export const initState: ModelSettings = linearModelDefinition.defaultSettings('regression');

export const useModelSettingsStore = create<ModelSettings>(() => initState);
