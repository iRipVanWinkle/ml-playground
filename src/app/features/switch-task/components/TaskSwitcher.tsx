import { EnhancedTabs } from '@/app/shared/ui';
import type { TaskType } from '@/app/shared/types';
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

    return (
        <EnhancedTabs
            defaultValue={taskType}
            className="w-full"
            onValueChange={handleTaskTypeChange}
        >
            <EnhancedTabs.List variant="underline" data-testid="task-switcher-list">
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
