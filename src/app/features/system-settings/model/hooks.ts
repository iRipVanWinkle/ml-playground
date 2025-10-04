import { updateBackend, updateRandomSeed } from './actions';
import { useSystemSettings } from './store';

export const useBackend = () => {
    const backend = useSystemSettings((state) => state.backend);

    return [backend, updateBackend] as const;
};

export const useRandomSeed = () => {
    const randomSeed = useSystemSettings((state) => state.randomSeed);

    return [randomSeed, updateRandomSeed] as const;
};
