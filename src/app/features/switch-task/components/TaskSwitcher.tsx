import { useMemo } from 'react';
import { EnhancedTabs } from '@/app/shared/ui';
import type { TaskType } from '@/app/shared/types';
import { getModelRegistry } from '@/app/models/ui-registry';
import { TASK_TYPES } from '../constants';
import { useTaskType } from '../store/hooks';
import { setTaskType } from '../store/actions';

type TaskSwitcherProps = {
    disabled?: boolean;
    onChange: (taskType: TaskType) => void;
};

export function TaskSwitcher({ disabled, onChange }: TaskSwitcherProps) {
    const taskType = useTaskType();

    const handleTaskTypeChange = (taskType: string) => {
        setTaskType(taskType as TaskType);
        onChange(taskType as TaskType);
    };

    const availableTaskTypes = useMemo(() => {
        const registry = getModelRegistry();

        return TASK_TYPES.filter((tt) => {
            const modelDefinitions = registry.getForTask(tt.value);

            return modelDefinitions.length > 0;
        });
    }, []);

    return (
        <EnhancedTabs
            defaultValue={taskType}
            onValueChange={handleTaskTypeChange}
            variant="underline"
            scrollable
        >
            <EnhancedTabs.List data-testid="task-switcher-list">
                {availableTaskTypes.map((tt) => (
                    <EnhancedTabs.Trigger
                        key={tt.value}
                        value={tt.value}
                        disabled={disabled}
                        icon={tt.icon}
                    >
                        {tt.label}
                    </EnhancedTabs.Trigger>
                ))}
            </EnhancedTabs.List>
        </EnhancedTabs>
    );
}
