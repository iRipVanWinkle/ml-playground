import { switchTask, useTaskType } from '@/app/store';
import { InlineSelect, Section, StepNum } from '../../../shared';
import type { TaskType } from '@/app/shared/types';
import { getModelRegistry } from '@/app/models/ui-registry';

type TaskTypeObject = {
    value: TaskType;
    label: string;
    hint: string;
    description: React.ReactNode;
};

const TASK_TYPES = [
    {
        value: 'regression',
        label: 'predict a number',
        hint: 'regression',
        description: (
            <>
                Predicts a <strong>continuous number</strong>. The model learns a function <code>ŷ = f(x)</code> mapping input features to a real value, and training tunes it to keep predictions close to the true numbers on average. Use it when the answer is "how much" or "how many."
            </>
        ),
    },
    {
        value: 'classification',
        label: 'predict a category',
        hint: 'classification',
        description: (
            <>
                Predicts <strong>which category</strong> something belongs to, from a fixed, known list. The model scores every class and picks the most likely one; training pushes that score toward the correct label. Use it when the answer is "which one."
            </>
        ),
    },
    {
        value: 'clustering',
        label: 'find natural groups',
        hint: 'clustering',
        description: (
            <>
                Groups similar rows together <strong>without using labels</strong>. There's no ground truth to check against: the algorithm proposes structure, and you decide whether the groups mean anything. Use it to explore data, not to predict.
            </>
        ),
    },
    {
        value: 'anomaly',
        label: 'flag the unusual ',
        hint: 'anomaly detection',
        description: (
            <>
                Learns <strong>what normal looks like</strong> and flags points that don't fit. You usually have plenty of normal examples and few or no labeled anomalies, so it's a one-class problem rather than a two-class one. Use it when the answer is "is this unusual?"
            </>
        ),
    },
] as TaskTypeObject[];

export function TaskSection() {
    const taskType = useTaskType();

    const handleTaskTypeChange = (taskType: string) => {
        switchTask(taskType as TaskType);
    };

    const registry = getModelRegistry();
    const availableTaskTypes = TASK_TYPES.filter((tt) => {
        const modelDefinitions = registry.getForTask(tt.value);

        return modelDefinitions.length > 0;
    });

    const selectedTaskType = TASK_TYPES.find((tt) => tt.value === taskType);

    return (
        <Section step={1} total={7}>
            <Section.Header>
                <StepNum />
                <Section.Title>Choose a task</Section.Title>
            </Section.Header>
            <Section.Body>
                <p className="text-xs">
                    What kind of problem are we solving? The task type determines which models,
                    metrics, and preprocessing steps make sense.
                </p>
                <p className="text-2xl font-light">
                    I want to{' '}<InlineSelect value={taskType} onValueChange={handleTaskTypeChange}>
                        <InlineSelect.Trigger placeholder="pick a task" />
                        <InlineSelect.Content>
                            {availableTaskTypes.map((tt) => (
                                <InlineSelect.Item key={tt.value} value={tt.value} hint={tt.hint}>
                                    {tt.label}
                                </InlineSelect.Item>
                            ))}
                        </InlineSelect.Content>
                    </InlineSelect>
                    {' '} — {selectedTaskType?.hint}
                </p>
                <p>
                    {selectedTaskType?.description}
                </p>
            </Section.Body>
        </Section>
    );
}
