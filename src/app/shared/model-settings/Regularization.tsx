import { Field, Input, Select, Slider } from '@/app/shared/ui';
import type { RegularizationType, RegularizationConfig } from '@/ml/factories';

type RegularizationProps = {
    regularization: RegularizationConfig;
    disabled?: boolean;
    onChange: (config: RegularizationConfig) => void;
};

const DEFAULT_LAMBDA = 1;
const DEFAULT_ELASTICNET_ALPHA = 0.5;
const REGULARIZATION_INFO =
    'Prevents the model from overfitting to training data. Helps it work on new data.';
const LAMBDA_INFO =
    'Controls the strength of regularization. Higher values make the model simpler.';
const ELASTICNET_ALPHA_INFO =
    'Controls the balance between ignoring features and keeping them small.';

const DEFAULT_REGULARIZATIONS = [
    {
        value: 'none',
        label: 'None',
    },
    {
        value: 'l2',
        label: 'L2 (Ridge)',
        info: 'Keeps model values small and spread evenly. Prevents the model from relying too much on any single feature.',
    },
    {
        value: 'l1',
        label: 'L1 (Lasso)',
        info: 'Sets some weights to exactly zero, which helps ignore less important features. Useful when you have many features.',
    },
    {
        value: 'elasticnet',
        label: 'Elastic Net',
        info: 'Combines L1 and L2 regularization. Can both ignore unimportant features and keep remaining weights small.',
    },
];

export default function Regularization({
    regularization,
    disabled,
    onChange,
}: RegularizationProps) {
    const handleFunctionChange = (type: RegularizationType) => {
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
                <Field
                    label="Regularization"
                    htmlFor="regularizationSelect"
                    info={REGULARIZATION_INFO}
                >
                    <Select
                        disabled={disabled}
                        value={regularization.type as string}
                        onValueChange={(value) => handleFunctionChange(value as RegularizationType)}
                    >
                        <Select.Trigger
                            id="regularizationSelect"
                            className="w-full truncate"
                            data-testid="regularization-select"
                        >
                            <Select.Value placeholder="Select regularization method" />
                        </Select.Trigger>
                        <Select.Content>
                            {DEFAULT_REGULARIZATIONS.map((func) => (
                                <Select.Item key={func.value} value={func.value} title={func.info}>
                                    {func.label}
                                </Select.Item>
                            ))}
                        </Select.Content>
                    </Select>
                </Field>

                {(isL || isElasticNet) && (
                    <Field label="Lambda" htmlFor="lambdaInput" info={LAMBDA_INFO}>
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
                <Field label="Alpha (α)" htmlFor="alphaInput" info={ELASTICNET_ALPHA_INFO}>
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
