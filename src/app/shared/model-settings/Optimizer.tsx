import { Checkbox, Field, InfoTooltip, Input, Label, Select } from '@/app/shared/ui';
import type { OptimizerConfig } from '@/ml/factories';

type OptimizerProps = {
    optimizer: OptimizerConfig;
    disabled?: boolean;
    onChange: (config: OptimizerConfig) => void;
};

const OPTIMIZER_INFO = 'The method that adjusts model parameters to reduce loss during training.';
const MAX_ITERATIONS_INFO = 'Maximum training steps before stopping.';
const TOLERANCE_INFO = 'Stops early if improvement is smaller than this value.';
const LEARNING_RATE_INFO = 'How big a step the optimizer takes when updating the model.';
const SCHEDULER_INFO = 'Gradually reduces learning rate for finer adjustments.';
const DECAY_OFFSET_INFO = 'Smooths the decay curve. Higher values mean slower, gentler decay.';
const DECAY_POWER_INFO = 'Controls the steepness of the decay. Higher values mean faster decay.';

const DEFAULT_OPTIMIZER_TYPES = [
    {
        value: 'batch',
        label: 'Batch Gradient Descent',
        info: 'Uses all data for each update. Stable but can be slow on large datasets.',
    },
    {
        value: 'sgd',
        label: 'Stochastic Gradient Descent',
        info: 'Updates after each small batch. Faster but less stable than using all data.',
    },
    {
        value: 'momentum',
        label: 'Momentum',
        info: 'Remembers past updates to speed up learning.',
    },
    {
        value: 'adam',
        label: 'Adam',
        info: 'An adaptive method that combines momentum and per-parameter learning rates.',
    },
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
            <Field label="Optimizer" htmlFor="optimizerSelect" info={OPTIMIZER_INFO}>
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
                            <Select.Item
                                key={option.value}
                                value={option.value}
                                title={option.info}
                            >
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
                <Field
                    label="Max Iterations"
                    htmlFor="maxIterationsInput"
                    info={MAX_ITERATIONS_INFO}
                >
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

                <Field label="Tolerance" htmlFor="toleranceInput" info={TOLERANCE_INFO}>
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

            <Field label="Learning Rate" htmlFor="learningRateInput" info={LEARNING_RATE_INFO}>
                <div className="grid grid-cols-2 gap-2">
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

                    <div className="flex items-center gap-2">
                        <Checkbox
                            id="schedulerCheckbox"
                            data-testid="scheduler-checkbox"
                            disabled={disabled}
                            checked={!!optimizer.scheduler}
                            onCheckedChange={(checked) => handleSchedulerChange(checked === true)}
                        />
                        <div className="flex items-center gap-1">
                            <Label htmlFor="schedulerCheckbox">Enable scheduler</Label>
                            <InfoTooltip>{SCHEDULER_INFO}</InfoTooltip>
                        </div>
                    </div>
                </div>
            </Field>

            {optimizer.scheduler && (
                <div className="grid grid-cols-2 gap-2">
                    <Field
                        label="Decay Offset (s₀)"
                        htmlFor="decayOffsetInput"
                        info={DECAY_OFFSET_INFO}
                    >
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
                    <Field
                        label="Decay Power (p)"
                        htmlFor="decayPowerInput"
                        info={DECAY_POWER_INFO}
                    >
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
