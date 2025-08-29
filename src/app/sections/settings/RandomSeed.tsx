import { Field } from '@/app/components/ui/field';
import { Input } from '@/app/components/ui/input';

type RandomSeedProps = {
    value?: number;
    disabled: boolean;
    onChange: (value?: number) => void;
};

export function RandomSeed({ value, disabled, onChange }: RandomSeedProps) {
    const handleChange = (newValue?: string) => {
        onChange(newValue ? Number(newValue) : undefined);
    };

    return (
        <Field label="Random Seed" htmlFor="random-seed">
            <Input
                id="random-seed"
                className="w-50"
                type="number"
                value={value ?? ''}
                disabled={disabled}
                onChange={(e) => handleChange(e.target.value)}
            />
        </Field>
    );
}
