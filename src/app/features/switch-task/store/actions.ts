import type { TaskType } from '@/app/shared/types';
import { useTaskSwitcherStore } from './store';

export function setTaskType(taskType: TaskType) {
    useTaskSwitcherStore.setState({ taskType });
}
