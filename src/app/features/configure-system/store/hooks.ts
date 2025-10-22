import { useSystemStore } from './store';

export const useBackend = () => useSystemStore((store) => store.backend);
export const useRandomSeed = () => useSystemStore((store) => store.randomSeed);
