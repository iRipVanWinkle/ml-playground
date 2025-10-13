import { create } from 'zustand';
import type { SystemSettings } from './types';

const initState: SystemSettings = {
    backend: 'auto',
    randomSeed: 42,
};

export const useSystemStore = create(() => initState);
