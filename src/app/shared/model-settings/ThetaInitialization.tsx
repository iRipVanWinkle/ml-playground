import { Field, Input, Select } from '@/app/shared/ui';
import type { ThetaInitializationConfig } from '@/ml/factories';

type InitThetaProps = {
    thetaInitialization: ThetaInitializationConfig;
    disabled?: boolean;
    onChange: (thetaInitialization: ThetaInitializationConfig) => void;
};

const DEFAULT_INITIALIZER_CONFIGS = {
    zeros: { type: 'zeros' },
    ones: { type: 'ones' },
    constant: { type: 'constant', value: 0.5 },
    uniform: { type: 'uniform', min: -0.05, max: 0.05 },
    normal: { type: 'normal', mean: 0, stddev: 0.05 },
    xavierUniform: { type: 'xavierUniform' },
    xavierNormal: { type: 'xavierNormal' },
    heUniform: { type: 'heUniform' },
    heNormal: { type: 'heNormal' },
} as Record<ThetaInitializationConfig['type'], ThetaInitializationConfig>;

const DEFAULT_THETA_INITIALIZERS = [
    { value: 'zeros', label: 'Zeros' },
    { value: 'ones', label: 'Ones' },
    { value: 'constant', label: 'Constant' },
    { value: 'uniform', label: 'Random Uniform' },
    { value: 'normal', label: 'Random Normal (Gaussian)' },
    { value: 'xavierUniform', label: 'Xavier / Glorot Uniform' },
    { value: 'xavierNormal', label: 'Xavier / Glorot Normal' },
    { value: 'heUniform', label: 'He Uniform' },
    { value: 'heNormal', label: 'He Normal' },
];

export default function ThetaInitialization({
    thetaInitialization,
    disabled,
    onChange,
}: InitThetaProps) {
    const handleTypeChange = (type: ThetaInitializationConfig['type']) => {
        onChange({ ...DEFAULT_INITIALIZER_CONFIGS[type] });
    };

    const handleInputChange = (key: 'value' | 'min' | 'max' | 'mean' | 'stddev', value: string) => {
        onChange({ ...thetaInitialization, [key]: Number(value) });
    };

    return (
        <>
            <Field label="Weight Initialization" htmlFor="thetaInitializationSelect">
                <Select
                    disabled={disabled}
                    value={thetaInitialization.type}
                    onValueChange={(value) =>
                        handleTypeChange(value as ThetaInitializationConfig['type'])
                    }
                >
                    <Select.Trigger
                        id="thetaInitializationSelect"
                        className="w-full truncate"
                        data-testid="theta-initialization-select"
                    >
                        <Select.Value placeholder="Select initial weight" />
                    </Select.Trigger>
                    <Select.Content>
                        {DEFAULT_THETA_INITIALIZERS.map((initializer) => (
                            <Select.Item key={initializer.value} value={initializer.value}>
                                {initializer.label}
                            </Select.Item>
                        ))}
                    </Select.Content>
                </Select>
            </Field>

            {thetaInitialization.type === 'constant' && (
                <div className="grid grid-cols-2 gap-2">
                    <Field label="Constant Value" htmlFor="constantValueInput">
                        <Input
                            id="constantValueInput"
                            data-testid="constant-value-input"
                            disabled={disabled}
                            type="number"
                            step={0.1}
                            value={thetaInitialization.value}
                            onChange={(e) => handleInputChange('value', e.target.value)}
                        />
                    </Field>
                </div>
            )}

            {thetaInitialization.type === 'uniform' && (
                <div className="grid grid-cols-2 gap-2">
                    <Field label="Min Value" htmlFor="uniformMinInput">
                        <Input
                            id="uniformMinInput"
                            disabled={disabled}
                            type="number"
                            step={0.01}
                            value={thetaInitialization.min}
                            onChange={(e) => handleInputChange('min', e.target.value)}
                        />
                    </Field>

                    <Field label="Max Value" htmlFor="uniformMaxInput">
                        <Input
                            id="uniformMaxInput"
                            disabled={disabled}
                            type="number"
                            step={0.01}
                            value={thetaInitialization.max}
                            onChange={(e) => handleInputChange('max', e.target.value)}
                        />
                    </Field>
                </div>
            )}

            {thetaInitialization.type === 'normal' && (
                <div className="grid grid-cols-2 gap-2">
                    <Field label="Mean" htmlFor="normalMeanInput">
                        <Input
                            id="normalMeanInput"
                            disabled={disabled}
                            type="number"
                            step={0.01}
                            value={thetaInitialization.mean}
                            onChange={(e) => handleInputChange('mean', e.target.value)}
                        />
                    </Field>

                    <Field label="Standard Deviation" htmlFor="normalStddevInput">
                        <Input
                            id="normalStddevInput"
                            disabled={disabled}
                            type="number"
                            step={0.01}
                            value={thetaInitialization.stddev}
                            onChange={(e) => handleInputChange('stddev', e.target.value)}
                        />
                    </Field>
                </div>
            )}
        </>
    );
}
