import { Field, Input } from '@/app/shared/ui';
import { setRandomSeed, useRandomSeed } from '@/app/store';

type RandomSeedInputProps = {
    disabled: boolean;
};

export function RandomSeedInput({ disabled }: RandomSeedInputProps) {
    const value = useRandomSeed();

    const handleChange = (newValue?: string) => {
        setRandomSeed(newValue ? Number(newValue) : undefined);
    };

    return (
        <Field label="Random Seed" htmlFor="randomSeedInput">
            <Input
                id="randomSeedInput"
                name="randomSeed"
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
