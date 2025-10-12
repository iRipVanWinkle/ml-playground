import { useSystemSettings } from './store';

export const useBackend = () => useSystemSettings((state) => state.backend);
export const useRandomSeed = () => useSystemSettings((state) => state.randomSeed);
