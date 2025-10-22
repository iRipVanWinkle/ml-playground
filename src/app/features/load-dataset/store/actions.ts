import type { Dataset } from '@/app/shared/types';
import { useDatasetStore, initState } from './store';

export function reset() {
    useDatasetStore.setState(initState, true);
}

export function setDataset(dataset: Dataset) {
    useDatasetStore.setState({ dataset });
}
