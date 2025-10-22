import { useSystemStore } from './store';
import type { SystemSettings } from './types';

export function updateBackend(backend: SystemSettings['backend']) {
    useSystemStore.setState({ backend });
}

export function updateRandomSeed(randomSeed: SystemSettings['randomSeed']) {
    useSystemStore.setState({ randomSeed });
}
