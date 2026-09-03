import { Section, StepNum } from '../../../shared';

export function VisualizeSection() {
    return (
        <Section step={6} total={7}>
            <Section.Header>
                <StepNum />
                <Section.Title>What did it learn?</Section.Title>
            </Section.Header>
            <Section.Body>
                <p>
                    Look inside the trained model to see which features mattered most and how it
                    is making its decisions.
                </p>
            </Section.Body>
        </Section>
    );
}
