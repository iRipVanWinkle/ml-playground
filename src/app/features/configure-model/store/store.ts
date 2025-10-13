import { create } from 'zustand';
import type { ModelSettings } from './types';
import { modelSettingsDefaults } from '../defaults';

export const initState: ModelSettings = modelSettingsDefaults['linear']('regression');

export const useModelSettingsStore = create(() => initState);
