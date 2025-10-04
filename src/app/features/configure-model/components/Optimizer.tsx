import { Checkbox, Field, Input, Label, Select } from '@/app/shared/ui';
import type { OptimizerConfig } from '@/app/store';

type OptimizerProps = {
    optimizer: OptimizerConfig;
    disabled?: boolean;
    onChange: (config: OptimizerConfig) => void;
};

const DEFAULT_OPTIMIZER_TYPES = [
    { value: 'batch', label: 'Batch Gradient Descent' },
    { value: 'sgd', label: 'Stochastic Gradient Descent' },
    { value: 'momentum', label: 'Momentum' },
    { value: 'adam', label: 'Adam' },
];

const DEFAULT_OPTIMIZER_CONFIGS = {
    batch: { type: 'batch' },
    sgd: { type: 'sgd', batchSize: 32 },
    momentum: { type: 'momentum', beta: 0.9 },
    adam: { type: 'adam', beta1: 0.9, beta2: 0.999 },
} as Record<OptimizerConfig['type'], OptimizerConfig>;

export default function Optimizer({ optimizer, disabled, onChange }: OptimizerProps) {
    // Handle optimizer type change
    const handleTypeChange = (type: string) => {
        const { maxIterations, tolerance, learningRate, scheduler, schedulerConfig } = optimizer;
        const optimizerType = type as OptimizerConfig['type'];

        const newConfig = {
            ...DEFAULT_OPTIMIZER_CONFIGS[optimizerType],
            maxIterations,
            tolerance,
            learningRate,
            scheduler,
            schedulerConfig,
        };

        onChange(newConfig);
    };

    // Handle input changes for optimizer parameters
    const handleInputChange = (
        key: keyof OptimizerConfig | 'batchSize' | 'beta' | 'beta1' | 'beta2',
        value: string,
    ) => {
        let preperedValue: number;
        if (key === 'batchSize' || key === 'maxIterations') {
            preperedValue = parseInt(value);
        } else {
            preperedValue = parseFloat(value);
        }
        const newConfig = { ...optimizer, [key]: preperedValue };
        onChange(newConfig as OptimizerConfig);
    };

    // Handle scheduler checkbox change
    const handleSchedulerChange = (checked: boolean) => {
        const newConfig = { ...optimizer, scheduler: checked };
        onChange(newConfig);
    };

    // Handle scheduler config changes
    const handleSchedulerConfigChange = (key: 's0' | 'p', value: string) => {
        const newConfig = {
            ...optimizer,
            schedulerConfig: {
                ...optimizer.schedulerConfig,
                [key]: value,
            },
        };
        onChange(newConfig);
    };

    return (
        <>
            <Field label="Optimizer" htmlFor="optimizerSelect">
                <Select disabled={disabled} value={optimizer.type} onValueChange={handleTypeChange}>
                    <Select.Trigger
                        id="optimizerSelect"
                        className="w-full truncate"
                        data-testid="optimizer-select"
                    >
                        <Select.Value placeholder="Select optimizer" />
                    </Select.Trigger>
                    <Select.Content>
                        {DEFAULT_OPTIMIZER_TYPES.map((option) => (
                            <Select.Item key={option.value} value={option.value}>
                                {option.label}
                            </Select.Item>
                        ))}
                    </Select.Content>
                </Select>
            </Field>

            {optimizer.type === 'sgd' && (
                <Field label="Batch Size" htmlFor="batchSizeInput">
                    <Input
                        id="batchSizeInput"
                        data-testid="batch-size-input"
                        className="w-1/2"
                        disabled={disabled}
                        step={1}
                        min={1}
                        type="number"
                        placeholder="Batch size"
                        value={optimizer.batchSize}
                        onChange={(e) => handleInputChange('batchSize', e.target.value)}
                    />
                </Field>
            )}

            {optimizer.type === 'momentum' && (
                <Field label="Beta (Momentum Factor)" htmlFor="momentumBetaInput">
                    <Input
                        id="momentumBetaInput"
                        className="w-1/2"
                        disabled={disabled}
                        type="number"
                        step={0.1}
                        min={0}
                        max={0.9999}
                        placeholder="Beta (momentum factor)"
                        value={optimizer.beta}
                        onChange={(e) => handleInputChange('beta', e.target.value)}
                    />
                </Field>
            )}

            {optimizer.type === 'adam' && (
                <div className="grid grid-cols-2 gap-2">
                    <Field label="Beta 1">
                        <Input
                            className="w-full"
                            disabled={disabled}
                            type="number"
                            step={0.01}
                            min={0}
                            max={0.99}
                            placeholder="Beta 1"
                            value={optimizer.beta1}
                            onChange={(e) => handleInputChange('beta1', e.target.value)}
                        />
                    </Field>
                    <Field label="Beta 2">
                        <Input
                            className="w-full"
                            disabled={disabled}
                            type="number"
                            step={0.01}
                            min={0}
                            max={0.9999}
                            placeholder="Beta 2"
                            value={optimizer.beta2}
                            onChange={(e) => handleInputChange('beta2', e.target.value)}
                        />
                    </Field>
                </div>
            )}

            <div className="grid grid-cols-2 gap-2">
                <Field label="Max Iterations" htmlFor="maxIterationsInput">
                    <Input
                        id="maxIterationsInput"
                        data-testid="max-iterations-input"
                        disabled={disabled}
                        type="number"
                        placeholder="Max Iterations"
                        value={optimizer.maxIterations}
                        onChange={(e) => handleInputChange('maxIterations', e.target.value)}
                    />
                </Field>

                <Field label="Tolerance" htmlFor="toleranceInput">
                    <Input
                        id="toleranceInput"
                        disabled={disabled}
                        type="number"
                        min={0}
                        step={0.0001}
                        placeholder="Tolerance"
                        value={optimizer.tolerance}
                        onChange={(e) => handleInputChange('tolerance', e.target.value)}
                    />
                </Field>
            </div>
            <div className="grid grid-cols-2 gap-2">
                <Field label="Learning Rate" htmlFor="learningRateInput">
                    <Input
                        id="learningRateInput"
                        data-testid="learning-rate-input"
                        disabled={disabled}
                        type="number"
                        step={0.001}
                        min={0}
                        placeholder="Alpha"
                        value={optimizer.learningRate}
                        onChange={(e) => handleInputChange('learningRate', e.target.value)}
                    />
                </Field>
                <div className="flex items-center gap-2 mt-6">
                    <Checkbox
                        id="schedulerCheckbox"
                        data-testid="scheduler-checkbox"
                        disabled={disabled}
                        checked={!!optimizer.scheduler}
                        onCheckedChange={(checked) => handleSchedulerChange(checked === true)}
                    />
                    <Label htmlFor="schedulerCheckbox">Enable scheduler</Label>
                </div>
            </div>

            {optimizer.scheduler && (
                <div className="grid grid-cols-2 gap-2">
                    <Field label="Decay Offset (s₀)" htmlFor="decayOffsetInput">
                        <Input
                            id="decayOffsetInput"
                            data-testid="decay-offset-input"
                            disabled={disabled}
                            type="number"
                            step={0.1}
                            min={0}
                            value={optimizer.schedulerConfig.s0}
                            onChange={(e) => handleSchedulerConfigChange('s0', e.target.value)}
                        />
                    </Field>
                    <Field label="Decay Power (p)" htmlFor="decayPowerInput">
                        <Input
                            id="decayPowerInput"
                            data-testid="decay-power-input"
                            disabled={disabled}
                            type="number"
                            step={0.1}
                            min={0}
                            value={optimizer.schedulerConfig.p}
                            onChange={(e) => handleSchedulerConfigChange('p', e.target.value)}
                        />
                    </Field>
                </div>
            )}
        </>
    );
}
