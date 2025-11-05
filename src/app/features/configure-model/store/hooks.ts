import { setModelType } from './actions';
import { useModelSettingsStore } from './store';

export const useModelType = () => useModelSettingsStore((state) => state.type);

export const useSetModelType = () => setModelType;
