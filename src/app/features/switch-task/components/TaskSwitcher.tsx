import { useMemo } from 'react';
import { EnhancedTabs } from '@/app/shared/ui';
import type { TaskType } from '@/app/shared/types';
import { getModelRegistry } from '@/app/models/ui-registry';
import { TASK_TYPES } from '../constants';
import { switchTask, useIsTraining, useTaskType } from '@/app/store';

type TaskSwitcherProps = {
    disabled?: boolean;
};

export function TaskSwitcher({ disabled }: TaskSwitcherProps) {
    const taskType = useTaskType();
    const isTraining = useIsTraining();

    const handleTaskTypeChange = (taskType: string) => {
        switchTask(taskType as TaskType);
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
                        disabled={disabled ?? isTraining}
                        icon={tt.icon}
                    >
                        {tt.label}
                    </EnhancedTabs.Trigger>
                ))}
            </EnhancedTabs.List>
        </EnhancedTabs>
    );
}
