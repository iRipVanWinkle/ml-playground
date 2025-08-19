import { Checkbox } from '@/app/components/ui/checkbox';
import { Field } from '@/app/components/ui/field';
import { Input } from '@/app/components/ui/input';
import { Label } from '@/app/components/ui/label';
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from '@/app/components/ui/select';
import type { OptimizerConfig } from '@/app/store';
import type { OptionList } from '../types';

type OptimizerProps = {
    optimizer: OptimizerConfig;
    disabled?: boolean;
    onChange: (config: OptimizerConfig) => void;
};

const DEFAULT_OPTIMIZER_TYPES: OptionList = [
    { value: 'batch', label: 'Batch Gradient Descent' },
    { value: 'sgd', label: 'Stochastic Gradient Descent' },
    { value: 'momentum', label: 'Momentum' },
];

const DEFAULT_OPTIMIZER_CONFIGS = {
    batch: { type: 'batch' },
    sgd: { type: 'sgd', batchSize: 32 },
    momentum: { type: 'momentum', beta: 0.9 },
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
        key: keyof OptimizerConfig | 'batchSize' | 'beta',
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
            <Field label="Optimizer">
                <Select disabled={disabled} value={optimizer.type} onValueChange={handleTypeChange}>
                    <SelectTrigger className="w-full truncate">
                        <SelectValue placeholder="Select optimizer" />
                    </SelectTrigger>
                    <SelectContent>
                        {DEFAULT_OPTIMIZER_TYPES.map((option) => (
                            <SelectItem
                                key={option.value}
                                value={option.value}
                                disabled={option.disabled}
                            >
                                {option.label}
                            </SelectItem>
                        ))}
                    </SelectContent>
                </Select>
            </Field>

            {optimizer.type === 'sgd' && (
                <Field label="Batch Size">
                    <Input
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
                <Field label="Beta (Momentum Factor)">
                    <Input
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

            <div className="grid grid-cols-2 gap-2">
                <Field label="Max iterations">
                    <Input
                        disabled={disabled}
                        type="number"
                        placeholder="Max Iterations"
                        value={optimizer.maxIterations}
                        onChange={(e) => handleInputChange('maxIterations', e.target.value)}
                    />
                </Field>

                <Field label="Tolerance">
                    <Input
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
                <Field label="Learning Rate">
                    <Input
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
                        id="scheduler"
                        disabled={disabled}
                        checked={!!optimizer.scheduler}
                        onCheckedChange={(checked) => handleSchedulerChange(checked === true)}
                    />
                    <Label htmlFor="scheduler">Enable scheduler</Label>
                </div>
            </div>

            {optimizer.scheduler && (
                <div className="grid grid-cols-2 gap-2">
                    <Field label="Decay Offset (s₀)">
                        <Input
                            disabled={disabled}
                            type="number"
                            step={0.1}
                            min={0}
                            value={optimizer.schedulerConfig.s0}
                            onChange={(e) => handleSchedulerConfigChange('s0', e.target.value)}
                        />
                    </Field>
                    <Field label="Decay Power (p)">
                        <Input
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
