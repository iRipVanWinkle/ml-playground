import { useTaskSwitcherStore } from './store';

export const useTaskType = () => useTaskSwitcherStore((state) => state.taskType);
