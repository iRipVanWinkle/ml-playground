import { Section, StepNum } from '../../../shared';

export function TrainingSection() {
    return (
        <Section step={5} total={7}>
            <Section.Header>
                <StepNum />
                <Section.Title>Train &amp; observe</Section.Title>
            </Section.Header>
            <Section.Body>
                <p>
                    Hit play to watch 140 iterations of batch gradient descent on your GPU. The loss curve shows train versus held-out validation — divergence means you're memorizing, not learning.
                </p>
            </Section.Body>
        </Section>
    );
}
