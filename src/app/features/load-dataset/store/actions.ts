import type { DataState } from './types';
import { useDatasetStore, initState } from './store';

export function reset() {
    useDatasetStore.setState(initState, true);
}

export function setDataset(data: DataState) {
    useDatasetStore.setState(data);
}
