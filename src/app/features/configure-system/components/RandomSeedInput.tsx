import { Field, Input } from '@/app/shared/ui';
import { useSystemStore } from '../store';
import { updateRandomSeed } from '../store/actions';

type RandomSeedInputProps = {
    disabled: boolean;
};

const useRandomSeed = () => useSystemStore((state) => state.randomSeed);

export function RandomSeedInput({ disabled }: RandomSeedInputProps) {
    const value = useRandomSeed();

    const handleChange = (newValue?: string) => {
        updateRandomSeed(newValue ? Number(newValue) : undefined);
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
