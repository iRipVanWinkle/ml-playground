import { useDatasetStore } from '@/app/features/load-dataset';
import { useSystemStore } from '@/app/features/system-settings';
import { useTransformationStore } from '@/app/features/transform-data';
import type { UseBoundStore, StoreApi } from 'zustand';

const stores = {
    settings: useSystemStore,
    transformations: useTransformationStore,
    dataset: useDatasetStore,
};

export type Stores = typeof stores;
export type StoreTypes = keyof typeof stores;

export type StoreStates = {
    [K in StoreTypes]: ReturnType<Stores[K]['getState']>;
};

export type StoreMap = {
    [K in StoreTypes]: UseBoundStore<StoreApi<StoreStates[K]>>;
};

class StoreManager {
    private stores: Stores = stores;

    getStore<K extends StoreTypes>(name: K): StoreMap[K] {
        return this.stores[name];
    }

    getState<K extends StoreTypes>(name: K): StoreStates[K] {
        return this.stores[name].getState() as StoreStates[K];
    }
}

export const storeManager = new StoreManager();
