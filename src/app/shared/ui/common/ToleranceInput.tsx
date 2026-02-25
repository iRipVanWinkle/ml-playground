import type { ChangeEvent, ComponentProps } from 'react';
import { InputGroup } from '../basic/input-group';
import { Switch } from '../basic/switch';

type ToleranceInputProps = Omit<ComponentProps<'input'>, 'onChange'> & {
    onChange: (value: number | undefined) => void;
};

const DEFAULT_TOLERANCE = 0.0001;
const DEFAULT_STEP = 0.0001;

export function ToleranceInput(props: ToleranceInputProps) {
    const { onChange, value, disabled, step, ...rest } = props;

    const handleInputChange = (e: ChangeEvent<HTMLInputElement>) => {
        onChange(e.target.value ? parseFloat(e.target.value) : undefined);
    };

    const handleToleranceToggle = (enabled: boolean) => {
        onChange(enabled ? DEFAULT_TOLERANCE : undefined);
    };
    return (
        <InputGroup>
            <InputGroup.Input
                disabled={disabled || value === undefined}
                placeholder="Off"
                step={step ?? DEFAULT_STEP}
                min={0}
                type="number"
                data-testid="tolerance-input"
                value={value ?? ''}
                onChange={(e) => handleInputChange(e)}
                {...rest}
            />
            <InputGroup.Addon align="inline-end">
                <Switch
                    id="tolerance-enabled"
                    checked={value !== undefined}
                    onCheckedChange={handleToleranceToggle}
                    disabled={disabled}
                    data-testid="tolerance-switch"
                />
            </InputGroup.Addon>
        </InputGroup>
    );
}
