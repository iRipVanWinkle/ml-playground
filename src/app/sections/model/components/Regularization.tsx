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
import type { OptionList } from '../../types';
import { Slider } from '@/app/components/ui/slider';

type RegularizationProps = {
    regularization: RegularizationConfig;
    disabled?: boolean;
    onChange: (config: RegularizationConfig) => void;
};

const DEFAULT_LAMBDA = 1;
const DEFAULT_ELASTICNET_ALPHA = 0.5;

const DEFAULT_REGULARIZATIONS: OptionList = [
    {
        value: 'none',
        label: 'None',
    },
    {
        value: 'l2',
        label: 'L2 (Ridge)',
    },
    {
        value: 'l1',
        label: 'L1 (Lasso)',
    },
    {
        value: 'elasticnet',
        label: 'Elastic Net',
    },
];

export default function Regularization({
    regularization,
    disabled,
    onChange,
}: RegularizationProps) {
    const handleFunctionChange = (type: RegularizationName) => {
        let lambda = undefined;
        let alpha = undefined;

        if (regularization.type !== 'none') {
            lambda = regularization.lambda;
        }

        if (regularization.type === 'elasticnet') {
            alpha = regularization.alpha;
        }

        if (type === 'none') {
            onChange({ type: 'none' });
        } else if (type === 'elasticnet') {
            onChange({
                type: 'elasticnet',
                lambda: lambda ?? DEFAULT_LAMBDA,
                alpha: alpha ?? DEFAULT_ELASTICNET_ALPHA,
            });
        } else {
            onChange({ type, lambda: lambda ?? DEFAULT_LAMBDA });
        }
    };

    const isL = regularization.type === 'l1' || regularization.type === 'l2';
    const isElasticNet = regularization.type === 'elasticnet';

    let containerClass = 'grid gap-2';
    if (isL || isElasticNet) {
        containerClass += ' grid-cols-2';
    }

    return (
        <>
            <div className={containerClass}>
                <Field label="Regularization" htmlFor="regularizationSelect">
                    <Select
                        disabled={disabled}
                        value={regularization.type as string}
                        onValueChange={(value) => handleFunctionChange(value as RegularizationName)}
                    >
                        <SelectTrigger
                            id="regularizationSelect"
                            className="w-full truncate"
                            data-testid="regularization-select"
                        >
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

                {(isL || isElasticNet) && (
                    <Field label="Lambda" htmlFor="lambdaInput">
                        <Input
                            id="lambdaInput"
                            data-testid="lambda-input"
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

            {isElasticNet && (
                <Field label="Alpha (α)" htmlFor="alphaInput">
                    <div className="flex justify-between">
                        <span className="text-xs text-muted-foreground">
                            L1 ({regularization.alpha.toFixed(1)})
                        </span>
                        <span className="text-xs text-muted-foreground">
                            L2 ({(1 - regularization.alpha).toFixed(1)})
                        </span>
                    </div>
                    <Slider
                        defaultValue={[regularization.alpha]}
                        max={1}
                        min={0}
                        step={0.1}
                        disabled={disabled}
                        onValueChange={(value) => onChange({ ...regularization, alpha: value[0] })}
                    />
                </Field>
            )}
        </>
    );
}
