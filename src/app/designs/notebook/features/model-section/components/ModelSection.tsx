import { Section, StepNum } from '../../../shared';

export function ModelSection() {
    return (
        <Section step={4} total={7}>
            <Section.Header>
                <StepNum />
                <Section.Title>Pick a model</Section.Title>
            </Section.Header>
            <Section.Body>
                <p>
                    Start with a preset — or hit Custom and tune every knob, in whichever view feels natural: a grouped form, or the model's own equation with each symbol editable in place.
                </p>
            </Section.Body>
        </Section>
    );
}
