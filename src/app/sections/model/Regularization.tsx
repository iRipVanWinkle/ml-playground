import { Field } from '@/app/components/ui/field';
import { Input } from '@/app/components/ui/input';
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from '@/app/components/ui/select';
import type { Regularization as RegularizationName, RegularizationConfig } from '@/app/store';
import type { OptionList } from '../types';

type RegularizationProps = {
    regularization: RegularizationConfig;
    disabled?: boolean;
    onChange: (config: RegularizationConfig) => void;
};

const DEFAULT_LAMBDA = 1;

const DEFAULT_REGULARIZATIONS: OptionList = [
    {
        value: 'none',
        label: 'None',
    },
    {
        value: 'l2',
        label: 'L2 (Ridge)',
    },
];

export default function Regularization({
    regularization,
    disabled,
    onChange,
}: RegularizationProps) {
    const handleFunctionChange = (type: RegularizationName) => {
        let lambda = undefined;

        if (regularization.type !== 'none') {
            lambda = regularization.lambda;
        }

        if (type === 'none') {
            onChange({ type: 'none' });
        } else {
            onChange({ type, lambda: lambda ?? DEFAULT_LAMBDA });
        }
    };

    const isL = regularization.type === 'l2';

    let containerClass = 'grid gap-2';
    if (isL) {
        containerClass += ' grid-cols-2';
    }

    return (
        <>
            <div className={containerClass}>
                <Field label="Regularization">
                    <Select
                        disabled={disabled}
                        value={regularization.type as string}
                        onValueChange={(value) => handleFunctionChange(value as RegularizationName)}
                    >
                        <SelectTrigger className="w-full truncate">
                            <SelectValue placeholder="Select regularization method" />
                        </SelectTrigger>
                        <SelectContent>
                            {DEFAULT_REGULARIZATIONS.map((func) => (
                                <SelectItem
                                    key={func.value}
                                    value={func.value}
                                    disabled={func.disabled}
                                >
                                    {func.label}
                                </SelectItem>
                            ))}
                        </SelectContent>
                    </Select>
                </Field>

                {isL && (
                    <Field label="Lambda">
                        <Input
                            disabled={disabled}
                            placeholder="Lambda (λ)"
                            step={0.1}
                            type="number"
                            value={regularization.lambda}
                            onChange={(e) =>
                                onChange({ ...regularization, lambda: parseFloat(e.target.value) })
                            }
                        />
                    </Field>
                )}
            </div>
        </>
    );
}
