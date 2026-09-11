import { setModelType, useModelType, useTaskType } from '@/app/store';
import { Bubble, BubbleGroup, Section, StepNum } from '../../../shared';
import { getModelRegistry } from '@/app/models/ui-registry';
import type { ModelType } from '@/app/models/types';

const modelRegistry = getModelRegistry();

export function ModelSection() {
    const taskType = useTaskType();
    const modelType = useModelType();

    const modelTypes = modelRegistry.getForTask(taskType);
    console.info(modelType, modelTypes);
    const handleChange = (value: string | null) => {
        if (value === null) {
            setModelType(modelTypes[0].key);
            return;
        }

        setModelType(value as ModelType);
    };

    return (
        <Section step={4} total={7}>
            <Section.Header>
                <StepNum />
                <Section.Title>Pick a model</Section.Title>
            </Section.Header>
            <Section.Body>
                <p>
                    Start with a preset - or hit Custom and tune every knob, in whichever view feels natural: a grouped form, or the model's own equation with each symbol editable in place.
                </p>

                <BubbleGroup value={modelType} onValueChange={handleChange} allowDeselect={false}>
                    <BubbleGroup.Label>Model</BubbleGroup.Label>

                    {modelTypes.map((option) => (
                        <Bubble key={option.key} value={option.key}>
                            {option.label}
                        </Bubble>
                    ))}
                </BubbleGroup>
            </Section.Body>
        </Section>
    );
}
