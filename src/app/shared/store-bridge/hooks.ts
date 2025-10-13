import { storeManager, type StoreStates, type StoreTypes } from './store-manager';

export function useManagedStore<K extends StoreTypes>(name: K) {
    const useStore = storeManager.getStore(name);

    return useStore();
}

export function useManagedSelector<K extends StoreTypes, T>(
    name: K,
    selector: (state: StoreStates[K]) => T,
) {
    const useStore = storeManager.getStore(name);

    return useStore(selector);
}
