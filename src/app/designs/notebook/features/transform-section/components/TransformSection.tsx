import { Section, StepNum } from '../../../shared';

export function TransformSection() {
    return (
        <Section step={3} total={7}>
            <Section.Header>
                <StepNum />
                <Section.Title>Shape the features</Section.Title>
            </Section.Header>
            <Section.Body>
                <p>
                    Without scaling, gradient descent chases the long, steep dimensions and ignores the short ones. Z-score puts every feature on comparable footing — mean 0, std 1, fit on train then applied to test.
                </p>
            </Section.Body>
        </Section>
    );
}
