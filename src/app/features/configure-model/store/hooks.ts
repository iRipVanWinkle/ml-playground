import { useModelSettingsStore } from './store';

export const useModelType = () => useModelSettingsStore((state) => state.type);
