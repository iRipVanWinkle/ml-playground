import { useSystemSettings } from './store';
import type { SystemSettings } from './types';

export function updateBackend(backend: SystemSettings['backend']) {
    useSystemSettings.setState({ backend });
}

export function updateRandomSeed(randomSeed: SystemSettings['randomSeed']) {
    useSystemSettings.setState({ randomSeed });
}
