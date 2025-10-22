import { EnhancedTabs } from '@/app/shared/ui';
import type { TaskType } from '@/app/shared/types';
import { TASK_TYPES } from '../constants';
import { useTaskType } from '../store/hooks';
import { setTaskType } from '../store/actions';

type TaskSwitcherProps = {
    disabled?: boolean;
};

export function TaskSwitcher({ disabled }: TaskSwitcherProps) {
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
