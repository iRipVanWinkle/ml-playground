import { Section, StepNum } from '../../../shared';

export function DatasetSection() {
    return (
        <Section step={2} total={7}>
            <Section.Header>
                <StepNum />
                <Section.Title>The dataset</Section.Title>
            </Section.Header>
            <Section.Body>
                <p>
                    Every model learns from examples. Here we look at the raw rows and columns
                    before any transformation happens.
                </p>
            </Section.Body>
        </Section>
    );
}
