import { useModelSettingsStore } from './store';

export const useClassificationType = () =>
    useModelSettingsStore((state) =>
        state.type === 'logistic' ? state.classificationType : undefined,
    );
