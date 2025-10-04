import { Field, Input } from '@/app/shared/ui';
import { useRandomSeed } from '../model/hooks';

type RandomSeedProps = {
    disabled: boolean;
};

export function RandomSeed({ disabled }: RandomSeedProps) {
    const [value, onChange] = useRandomSeed();

    const handleChange = (newValue?: string) => {
        onChange(newValue ? Number(newValue) : undefined);
    };

    return (
        <Field label="Random Seed" htmlFor="randomSeedInput">
            <Input
                id="randomSeedInput"
                data-testid="random-seed-input"
                className="w-50"
                type="number"
                value={value ?? ''}
                disabled={disabled}
                onChange={(e) => handleChange(e.target.value)}
            />
        </Field>
    );
}
