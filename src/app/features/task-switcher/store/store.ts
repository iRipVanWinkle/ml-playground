import { create } from 'zustand';
import type { TaskSwitcherStore } from './types';

export const initState: TaskSwitcherStore = {
    taskType: 'regression',
};

export const useTaskSwitcherStore = create(() => initState);
