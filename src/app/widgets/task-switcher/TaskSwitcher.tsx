import { setTaskType, useIsTraining, useTaskType, type TaskType } from '@/app/store';
import { EnhancedTabs } from '@/app/shared/ui';
import { TASK_TYPES } from './constants';

export function TaskSwitcher() {
    const isTraining = useIsTraining();
    const taskType = useTaskType();

    const handleTaskTypeChange = (taskType: string) => {
        setTaskType(taskType as TaskType);
    };

    return (
        <EnhancedTabs
            defaultValue={taskType}
            className="w-full"
            onValueChange={handleTaskTypeChange}
        >
            <EnhancedTabs.List variant="underline">
                {TASK_TYPES.map((tt) => (
                    <EnhancedTabs.Trigger
                        key={tt.value}
                        value={tt.value}
                        disabled={isTraining}
                        icon={tt.icon}
                    >
                        {tt.label}
                    </EnhancedTabs.Trigger>
                ))}
            </EnhancedTabs.List>
        </EnhancedTabs>
    );
}
