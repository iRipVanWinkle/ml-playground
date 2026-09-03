import type { NormalizationMethod } from '@/app/shared/types';
import { setNormalization, useNormalization } from '@/app/store';
import { Block, Bubble, BubbleGroup } from '../../../shared';

const NORMALIZATION_METHODS = [
    {
        value: 'zscore',
        label: 'Z-Score',
    },
    {
        value: 'linear',
        label: 'Min-Max',
    },
    {
        value: 'log',
        label: 'Log',
    },
] as const;

const NORMS: Record<string, { name: string; formula: string; desc: React.ReactNode }> = {
    zscore: {
        name: 'Z-score',
        formula: "x' = (x − μ) / σ",
        desc: (
            <>
                Centers each feature on its mean and rescales by its standard deviation, so the result has <em>mean 0</em> and <em>standard deviation 1</em>. Fit <code>μ</code> and <code>σ</code> on the training set only, then reuse those same values on test data — recomputing them would leak information from data the model shouldn't have seen.
            </>
        ),
    },
    linear: {
        name: 'Min-max',
        formula: "x' = (x − min) / (max − min)",
        desc: (
            <>
                Rescales each feature linearly into the <code>[0, 1] range</code>. It preserves the shape of the distribution, but it's sensitive to outliers: a single extreme value pushes the bounds out and squeezes every other point into a narrow band.
            </>
        ),
    },
    log: {
        name: 'Log',
        formula: 'x′ = ln(1 + x)',
        desc: (
            <>
                Compresses long right tails, which helps when a feature spans many orders of magnitude (income, population, prices). It needs non-negative inputs, and the 1 + keeps zeros from blowing up.
            </>
        ),
    },
};

export function NormalizationPicker() {
    const normalization = useNormalization();

    function handleNormalizationChange(value: string | null) {
        setNormalization(value as NormalizationMethod);
    }

    const selected = normalization ? NORMS[normalization] : undefined;

    return (
        <>
            <BubbleGroup value={normalization} onValueChange={handleNormalizationChange}>
                <BubbleGroup.Label>Normalize</BubbleGroup.Label>

                {NORMALIZATION_METHODS.map((option) => (
                    <Bubble key={option.value} value={option.value}>
                        {option.label}
                    </Bubble>
                ))}
            </BubbleGroup>

            {selected && (
                <Block>
                    <Block.Title>
                        <div className="text-sm font-bold tracking-tight text-foreground">{selected.name}</div>
                        <div className="mt-0.5 inline-block font-mono text-xs font-bold tracking-wide text-muted-foreground uppercase">
                            Normalization
                        </div>
                    </Block.Title>
                    <Block.Body>
                        <p className="mb-1.5 inline-block rounded-md border border-border bg-background px-2.5 py-1.5 font-mono text-sm tracking-wide text-foreground">
                            {selected.formula}
                        </p>
                        <p>{selected.desc}</p>
                    </Block.Body>
                </Block>
            )}
        </>
    );
}
