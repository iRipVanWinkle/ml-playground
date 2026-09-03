import { Section, StepNum } from '../../../shared';

export function UserExample() {
    return (
        <Section step={7} total={7}>
            <Section.Header>
                <StepNum />
                <Section.Title>Try it on a new example</Section.Title>
            </Section.Header>
            <Section.Body>
                <p>
                    Metrics describe the test set in aggregate — but the real feel for a model comes from poking it. Hand it a single fresh row and watch it commit to an answer. Every edit re-runs the forward pass instantly, so you can probe where the decision flips.
                </p>
            </Section.Body>
        </Section>
    );
}
