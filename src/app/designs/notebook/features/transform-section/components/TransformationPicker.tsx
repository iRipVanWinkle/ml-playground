import type { AnyTransformation, TransformationType } from '@/app/shared/types';
import { setTransformations, useTransformations } from '@/app/store';
import { Block, Bubble, BubbleGroup } from '../../../shared';
import { calculateTransformationOutputFeatures } from '@/app/features/transform-data/libs';

const TRANSFORMATION_TYPES = [
    {
        value: 'sinusoid',
        label: 'Sinusoid',
    },
    {
        value: 'cosinusoid',
        label: 'Cosinusoid',
    },
    {
        value: 'fourier',
        label: 'Fourier',
    },
    {
        value: 'polynomial',
        label: 'Polynomial',
    },
] as const;

const TRANSFORMATIONS: Record<
    TransformationType,
    { name: string; formula: (degree: number) => string; desc: (degree: number) => React.ReactNode }
> = {
    polynomial: {
        name: 'Polynomial',
        formula: (d) => d === 1 ? `[x]` : d === 2 ? `[x, x²]` : d === 3 ? `[x, x², x³]` : d === 4 ? `[x, x², x³, x⁴]` : `[x, x², x³, …, x^${d}]`,
        desc: (d) => (
            <>
                Adds powers of each feature up to degree <em>{d}</em>. Lets a linear model fit curves — but
                degree {'>'}3 risks oscillating between training points and overfitting hard.
            </>
        ),
    },
    fourier: {
        name: 'Fourier',
        formula: (d) => `[sin(kπx), cos(kπx)]  for k = 1…${d}`,
        desc: (d) => (
            <>
                Encodes each feature as <em>{d}</em> pairs of sine/cosine harmonics. Captures periodic
                structure (time-of-day, seasonality) that polynomials miss.
            </>
        ),
    },
    sinusoid: {
        name: 'Sinusoid',
        formula: (d) => d === 1 ? `[sin(πx)]` : d === 2 ? `[sin(πx), sin(2πx)]` : d === 3 ? `[sin(πx), sin(2πx), sin(3πx)]` : `[sin(πx), sin(2πx), …, sin(${d}πx)]`,
        desc: (d) => (
            <>
                Adds <em>{d}</em> sine basis functions per feature. Lighter than full Fourier when you only
                need odd-symmetric periodicity.
            </>
        ),
    },
    cosinusoid: {
        name: 'Cosinusoid',
        formula: (d) => d === 1 ? `[cos(πx)]` : d === 2 ? `[cos(πx), cos(2πx)]` : d === 3 ? `[cos(πx), cos(2πx), cos(3πx)]` : `[cos(πx), cos(2πx), …, cos(${d}πx)]`,
        desc: (d) => (
            <>
                Adds <em>{d}</em> cosine basis functions per feature. Useful for even-symmetric patterns and
                as a smooth, bounded alternative to polynomials.
            </>
        ),
    },
};

type TransformationPickerProps = {
    numFeatures: number;
};

export function TransformationPicker({ numFeatures }: TransformationPickerProps) {
    const transformations = useTransformations();
    const values = Array.from(new Set(transformations.map((t) => t.type)));

    function handleTransformationsChange(nextValues: string[]) {
        const kept = transformations.filter(
            (t) => t.type !== '' && nextValues.includes(t.type),
        );
        const added = nextValues
            .filter((type) => !kept.some((t) => t.type === type))
            .map((type) => ({ type: type as TransformationType, degree: 1 }));

        setTransformations([...kept, ...added]);
    }

    function handleDegreeChange(type: TransformationType, degree: number) {
        setTransformations(
            transformations.map((t) => (t.type === type ? { ...t, degree } : t)),
        );
    }

    const outputFeatures = calculateOutputFeatures(transformations, numFeatures);

    return (
        <>
            <BubbleGroup type="multiple" className="my-2" value={values} onValueChange={handleTransformationsChange}>
                <BubbleGroup.Label>Transformations</BubbleGroup.Label>

                {TRANSFORMATION_TYPES.map((option) => (
                    <TransformationBubble
                        key={option.value}
                        value={option.value}
                        label={option.label}
                        degree={transformations.find((t) => t.type === option.value)?.degree}
                        onDegreeChange={(degree) => handleDegreeChange(option.value, degree)}
                    />
                ))}
            </BubbleGroup>
            {transformations.length > 0 && <div className="my-2 text-xs">Expanded to {outputFeatures} features fed to the model.</div>}
            {transformations.map((t) => {
                const tx = TRANSFORMATIONS[t.type as TransformationType];
                if (!tx) return null;

                return (
                    <Block key={t.type}>
                        <Block.Title>
                            <div className="text-sm font-bold tracking-tight text-foreground">{tx.name}</div>
                            <div className="mt-0.5 inline-block font-mono text-2xs font-bold tracking-wide text-muted-foreground uppercase">
                                degree {t.degree}
                            </div>
                        </Block.Title>
                        <Block.Body>
                            <p className="mb-1.5 inline-block rounded-md border border-border bg-background px-2.5 py-1.5 font-mono text-sm tracking-wide text-foreground">
                                {tx.formula(t.degree)}
                            </p>
                            <p>{tx.desc(t.degree)}</p>
                        </Block.Body>
                    </Block>
                );
            })}
        </>
    );
}

type TransformationBubbleProps = {
    value: TransformationType;
    label: string;
    degree: number | undefined;
    onDegreeChange: (degree: number) => void;
};

function TransformationBubble({ value, label, degree, onDegreeChange }: TransformationBubbleProps) {
    return (
        <Bubble value={value}>
            {label}
            {degree !== undefined && (
                <Bubble.Counter value={degree} onValueChange={onDegreeChange} min={1} />
            )}
        </Bubble>
    );
}

function calculateOutputFeatures(transformations: AnyTransformation[], numFeatures: number): number {
    return transformations.reduce((sum, transformation) => {
        if (transformation.type === '') return sum;

        return sum + calculateTransformationOutputFeatures(
            transformation.type as TransformationType,
            transformation.degree,
            numFeatures
        );
    }, 0);
}
