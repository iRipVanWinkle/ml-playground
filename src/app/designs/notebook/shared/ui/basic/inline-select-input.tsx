import * as React from 'react';
import * as PopoverPrimitive from '@radix-ui/react-popover';
import * as SwitchPrimitive from '@radix-ui/react-switch';
import { ChevronDownIcon } from 'lucide-react';

import { cn } from '@/app/shared/ui/utils';

/**
 * An inline editable input that lives inside prose sentences: the trigger reads as
 * a dashed-underlined value, and clicking it opens a popover containing an input field
 * (such as a number input, range slider, or toggle switch), label, hint, and completion button.
 *
 * Number Input Example:
 * ```tsx
 * <InlineSelectInput value={seed} onValueChange={setSeed}>
 *     <InlineSelectInput.Trigger placeholder="random seed" />
 *     <InlineSelectInput.Content>
 *         <InlineSelectInput.Label>Random seed</InlineSelectInput.Label>
 *         <InlineSelectInput.Input type="number" min={0} max={9999} />
 *         <InlineSelectInput.Hint>Same seed → same shuffle → reproducible</InlineSelectInput.Hint>
 *         <InlineSelectInput.Footer>
 *             <InlineSelectInput.Done>Done</InlineSelectInput.Done>
 *         </InlineSelectInput.Footer>
 *     </InlineSelectInput.Content>
 * </InlineSelectInput>
 * ```
 *
 * Range Slider Example:
 * ```tsx
 * <InlineSelectInput value={splitPct} onValueChange={setSplitPct}>
 *     <InlineSelectInput.Trigger placeholder="split">
 *         {splitPct}/{100 - splitPct}
 *     </InlineSelectInput.Trigger>
 *     <InlineSelectInput.Content>
 *         <InlineSelectInput.Label>
 *             Train / test split — <b className="font-bold normal-case text-foreground">{splitPct}/{100 - splitPct}</b>
 *         </InlineSelectInput.Label>
 *         <InlineSelectInput.Range min={50} max={95} />
 *         <InlineSelectInput.Hint>
 *             Percent used for training; the rest is held out
 *         </InlineSelectInput.Hint>
 *         <InlineSelectInput.Footer>
 *             <InlineSelectInput.Done>Done</InlineSelectInput.Done>
 *         </InlineSelectInput.Footer>
 *     </InlineSelectInput.Content>
 * </InlineSelectInput>
 * ```
 *
 * Toggle Example:
 * ```tsx
 * <InlineSelectInput value={shuffled} onValueChange={setShuffled}>
 *     <InlineSelectInput.Trigger placeholder="shuffle">
 *         {shuffled ? 'shuffled' : 'unshuffled'}
 *     </InlineSelectInput.Trigger>
 *     <InlineSelectInput.Content>
 *         <InlineSelectInput.Label>Shuffle rows</InlineSelectInput.Label>
 *         <InlineSelectInput.Toggle onLabel="Shuffled" offLabel="Unshuffled" />
 *         <InlineSelectInput.Hint>Randomize row order before split</InlineSelectInput.Hint>
 *         <InlineSelectInput.Footer>
 *             <InlineSelectInput.Done>Done</InlineSelectInput.Done>
 *         </InlineSelectInput.Footer>
 *     </InlineSelectInput.Content>
 * </InlineSelectInput>
 * ```
 */

type InlineSelectInputValue = string | number | boolean;

interface InlineSelectInputContextValue {
    value?: InlineSelectInputValue;
    onValueChange?: (value: InlineSelectInputValue) => void;
    open: boolean;
    setOpen: (open: boolean) => void;
    disabled?: boolean;
}

const InlineSelectInputContext = React.createContext<InlineSelectInputContextValue | null>(null);

function useInlineSelectInputContext() {
    const context = React.useContext(InlineSelectInputContext);
    if (!context) {
        throw new Error(
            'InlineSelectInput compound components must be used within an InlineSelectInput',
        );
    }
    return context;
}

type InlineSelectInputProps<T extends InlineSelectInputValue = InlineSelectInputValue> = {
    value?: T;
    defaultValue?: T;
    onValueChange?: (value: T) => void;
    open?: boolean;
    defaultOpen?: boolean;
    onOpenChange?: (open: boolean) => void;
    disabled?: boolean;
    children?: React.ReactNode;
};

function InlineSelectInputRoot<T extends InlineSelectInputValue = InlineSelectInputValue>({
    value: valueProp,
    defaultValue,
    onValueChange,
    open: openProp,
    defaultOpen = false,
    onOpenChange,
    disabled = false,
    children,
}: InlineSelectInputProps<T>) {
    const [uncontrolledValue, setUncontrolledValue] = React.useState<InlineSelectInputValue>(
        defaultValue ?? '',
    );
    const isControlledValue = valueProp !== undefined;
    const value = isControlledValue ? valueProp : uncontrolledValue;

    const handleValueChange = (nextValue: InlineSelectInputValue) => {
        if (!isControlledValue) {
            setUncontrolledValue(nextValue);
        }
        onValueChange?.(nextValue as T);
    };

    const [uncontrolledOpen, setUncontrolledOpen] = React.useState(defaultOpen);
    const isControlledOpen = openProp !== undefined;
    const open = isControlledOpen ? openProp : uncontrolledOpen;

    const handleOpenChange = (nextOpen: boolean) => {
        if (disabled && nextOpen) return;
        if (!isControlledOpen) {
            setUncontrolledOpen(nextOpen);
        }
        onOpenChange?.(nextOpen);
    };

    const contextValue: InlineSelectInputContextValue = {
        value,
        onValueChange: handleValueChange,
        open,
        setOpen: handleOpenChange,
        disabled,
    };

    return (
        <PopoverPrimitive.Root
            data-slot="inline-select-input"
            open={open}
            onOpenChange={handleOpenChange}
        >
            <InlineSelectInputContext.Provider value={contextValue}>
                {children}
            </InlineSelectInputContext.Provider>
        </PopoverPrimitive.Root>
    );
}

type InlineSelectInputTriggerProps = React.ComponentProps<typeof PopoverPrimitive.Trigger> & {
    /** Shown while children is not provided or empty. */
    placeholder?: string;
    /** Whether to display the rotating chevron down icon. Defaults to true. */
    showChevron?: boolean;
    children?: React.ReactNode;
};

function InlineSelectInputTrigger({
    className,
    placeholder,
    showChevron = true,
    children,
    disabled: disabledProp,
    ...props
}: InlineSelectInputTriggerProps) {
    const { disabled: contextDisabled } = useInlineSelectInputContext();
    const disabled = disabledProp ?? contextDisabled;

    const hasChildren = children !== undefined && children !== null && children !== '';

    return (
        <PopoverPrimitive.Trigger asChild disabled={disabled} {...props}>
            <button
                type="button"
                data-slot="inline-select-input-trigger"
                className={cn(
                    'group/trigger inline cursor-pointer rounded-xs border-0 border-b border-dashed p-0 px-0.5 font-[inherit] text-[length:inherit] font-medium text-foreground italic',
                    'border-muted-foreground/60 bg-transparent box-decoration-clone transition-colors outline-none',
                    'hover:border-foreground data-[state=open]:border-foreground',
                    'focus-visible:ring-[3px] focus-visible:ring-ring/50 focus-visible:rounded-sm',
                    'data-placeholder:text-muted-foreground',
                    'disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:border-muted-foreground/60',
                    '[&_svg]:pointer-events-none [&_svg]:inline [&_svg]:size-3.5 [&_svg]:shrink-0 [&_svg]:align-middle [&_svg]:opacity-60 [&_svg]:ml-1',
                    '[&_svg]:transition-transform data-[state=open]:[&_svg]:rotate-180',
                    className,
                )}
                data-placeholder={!hasChildren ? '' : undefined}
            >
                {hasChildren ? (
                    children
                ) : (
                    <span className="not-italic text-muted-foreground">{placeholder}</span>
                )}
                {showChevron && <ChevronDownIcon aria-hidden />}
            </button>
        </PopoverPrimitive.Trigger>
    );
}

type InlineSelectInputContentProps = React.ComponentProps<typeof PopoverPrimitive.Content>;

function InlineSelectInputContent({
    className,
    children,
    align = 'start',
    sideOffset = 8,
    ...props
}: InlineSelectInputContentProps) {
    return (
        <PopoverPrimitive.Portal>
            <PopoverPrimitive.Content
                data-slot="inline-select-input-content"
                align={align}
                sideOffset={sideOffset}
                className={cn(
                    'relative z-50 min-w-[13rem] max-w-[18rem] origin-(--radix-popover-content-transform-origin) overflow-hidden rounded-xl border bg-popover p-2.5 text-sm font-normal tracking-normal text-popover-foreground not-italic shadow-md outline-none',
                    'data-[state=open]:animate-in data-[state=open]:fade-in-0 data-[state=open]:zoom-in-95',
                    'data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=closed]:zoom-out-95',
                    'data-[side=bottom]:slide-in-from-top-2 data-[side=top]:slide-in-from-bottom-2 data-[side=left]:slide-in-from-right-2 data-[side=right]:slide-in-from-left-2',
                    className,
                )}
                {...props}
            >
                {children}
            </PopoverPrimitive.Content>
        </PopoverPrimitive.Portal>
    );
}

type InlineSelectInputLabelProps = React.ComponentProps<'div'>;

function InlineSelectInputLabel({ className, ...props }: InlineSelectInputLabelProps) {
    return (
        <div
            data-slot="inline-select-input-label"
            className={cn(
                'mb-2 font-mono text-[10px] font-bold tracking-widest uppercase text-muted-foreground select-none',
                className,
            )}
            {...props}
        />
    );
}

type InlineSelectInputInputProps = React.ComponentProps<'input'>;

function InlineSelectInputInput({
    className,
    type = 'number',
    autoFocus = true,
    value: valueProp,
    onChange: onChangeProp,
    onKeyDown,
    ...props
}: InlineSelectInputInputProps) {
    const { value: contextValue, onValueChange, setOpen } = useInlineSelectInputContext();

    const inputValue = valueProp !== undefined ? valueProp : (contextValue ?? '');

    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        onChangeProp?.(e);
        const raw = e.target.value;
        if (type === 'number') {
            if (raw === '') {
                onValueChange?.('');
            } else {
                const num = Number(raw);
                onValueChange?.(Number.isNaN(num) ? raw : num);
            }
        } else {
            onValueChange?.(raw);
        }
    };

    const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
        onKeyDown?.(e);
        if (e.key === 'Enter' && !e.defaultPrevented) {
            e.preventDefault();
            setOpen(false);
        }
    };

    return (
        <input
            type={type}
            data-slot="inline-select-input-field"
            autoFocus={autoFocus}
            value={inputValue}
            onChange={handleChange}
            onKeyDown={handleKeyDown}
            className={cn(
                'flex h-8 w-full rounded-md border border-input bg-background/50 px-2.5 py-1 font-mono text-xs font-semibold text-foreground shadow-xs transition-colors',
                'focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px] focus-visible:outline-none',
                'placeholder:text-muted-foreground disabled:cursor-not-allowed disabled:opacity-50',
                className,
            )}
            {...props}
        />
    );
}

type InlineSelectInputRangeProps = React.ComponentProps<'input'>;

function InlineSelectInputRange({
    className,
    min = 0,
    max = 100,
    step = 1,
    value: valueProp,
    onChange: onChangeProp,
    autoFocus = false,
    ...props
}: InlineSelectInputRangeProps) {
    const { value: contextValue, onValueChange } = useInlineSelectInputContext();

    const rangeValue =
        valueProp !== undefined
            ? valueProp
            : contextValue !== undefined && contextValue !== ''
              ? Number(contextValue)
              : Number(min);

    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        onChangeProp?.(e);
        const num = Number(e.target.value);
        onValueChange?.(Number.isNaN(num) ? e.target.value : num);
    };

    return (
        <input
            type="range"
            data-slot="inline-select-input-range"
            min={min}
            max={max}
            step={step}
            value={rangeValue}
            onChange={handleChange}
            autoFocus={autoFocus}
            className={cn(
                'my-1 h-2 w-full cursor-pointer appearance-none rounded-lg bg-secondary accent-primary transition-opacity focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50',
                className,
            )}
            {...props}
        />
    );
}

type InlineSelectInputToggleProps = Omit<
    React.ComponentProps<typeof SwitchPrimitive.Root>,
    'value'
> & {
    /** Renders only the switch element without any wrapping label or row container. */
    standalone?: boolean;
    /** Visual style of the toggle. 'switch' displays a Radix switch, 'pills' displays segmented buttons. Defaults to 'switch'. */
    variant?: 'switch' | 'pills';
    /** Optional label or content displayed beside the toggle switch. */
    children?: React.ReactNode;
    /** Label shown when the switch is ON (used if children is not provided). */
    onLabel?: React.ReactNode;
    /** Label shown when the switch is OFF (used if children is not provided). */
    offLabel?: React.ReactNode;
    /** Value to emit when toggled ON. Defaults to true (or 'on' if current value is string). */
    trueValue?: InlineSelectInputValue;
    /** Value to emit when toggled OFF. Defaults to false (or 'off' if current value is string). */
    falseValue?: InlineSelectInputValue;
};

function InlineSelectInputToggle({
    className,
    checked: checkedProp,
    defaultChecked,
    onCheckedChange,
    disabled: disabledProp,
    standalone = false,
    variant = 'switch',
    children,
    onLabel,
    offLabel,
    trueValue,
    falseValue,
    ...props
}: InlineSelectInputToggleProps) {
    const {
        value: contextValue,
        onValueChange,
        disabled: contextDisabled,
    } = useInlineSelectInputContext();
    const disabled = disabledProp ?? contextDisabled;

    const isControlledChecked = checkedProp !== undefined;
    const isContextBoolean = typeof contextValue === 'boolean';
    const isContextString = typeof contextValue === 'string';

    const resolvedTrue = trueValue !== undefined ? trueValue : isContextString ? 'on' : true;
    const resolvedFalse = falseValue !== undefined ? falseValue : isContextString ? 'off' : false;

    const computeChecked = (): boolean => {
        if (isControlledChecked) return checkedProp;
        if (contextValue === undefined) return defaultChecked ?? false;
        if (isContextBoolean) return contextValue;
        if (isContextString) {
            return (
                contextValue === resolvedTrue || contextValue === 'true' || contextValue === 'on'
            );
        }
        if (typeof contextValue === 'number') {
            return contextValue !== 0;
        }
        return Boolean(contextValue);
    };

    const isChecked = computeChecked();

    const handleCheckedChange = (nextChecked: boolean) => {
        onCheckedChange?.(nextChecked);
        const nextValue = nextChecked ? resolvedTrue : resolvedFalse;
        onValueChange?.(nextValue);
    };

    if (variant === 'pills') {
        return (
            <div
                data-slot="inline-select-input-toggle-pills"
                className={cn('flex items-center gap-1.5 py-1', className)}
            >
                <button
                    type="button"
                    disabled={disabled}
                    onClick={() => handleCheckedChange(true)}
                    className={cn(
                        'inline-flex cursor-pointer items-center justify-center rounded-md px-3 py-1 font-mono text-xs font-semibold transition-colors',
                        isChecked
                            ? 'bg-foreground text-background shadow-xs'
                            : 'bg-muted/50 text-muted-foreground hover:bg-muted hover:text-foreground',
                        'focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none disabled:cursor-not-allowed disabled:opacity-50',
                    )}
                >
                    {onLabel ?? 'On'}
                </button>
                <button
                    type="button"
                    disabled={disabled}
                    onClick={() => handleCheckedChange(false)}
                    className={cn(
                        'inline-flex cursor-pointer items-center justify-center rounded-md px-3 py-1 font-mono text-xs font-semibold transition-colors',
                        !isChecked
                            ? 'bg-foreground text-background shadow-xs'
                            : 'bg-muted/50 text-muted-foreground hover:bg-muted hover:text-foreground',
                        'focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none disabled:cursor-not-allowed disabled:opacity-50',
                    )}
                >
                    {offLabel ?? 'Off'}
                </button>
            </div>
        );
    }

    const switchElement = (
        <SwitchPrimitive.Root
            data-slot="inline-select-input-toggle-switch"
            checked={isChecked}
            onCheckedChange={handleCheckedChange}
            disabled={disabled}
            className={cn(
                'peer inline-flex h-[1.15rem] w-8 shrink-0 cursor-pointer items-center rounded-full border border-transparent shadow-xs transition-all outline-none',
                'data-[state=checked]:bg-primary data-[state=unchecked]:bg-input dark:data-[state=unchecked]:bg-input/80',
                'focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px]',
                'disabled:cursor-not-allowed disabled:opacity-50',
                standalone && className,
            )}
            {...props}
        >
            <SwitchPrimitive.Thumb
                data-slot="inline-select-input-toggle-thumb"
                className={cn(
                    'pointer-events-none block size-4 rounded-full bg-background ring-0 transition-transform',
                    'dark:data-[state=unchecked]:bg-foreground dark:data-[state=checked]:bg-primary-foreground',
                    'data-[state=checked]:translate-x-[calc(100%-2px)] data-[state=unchecked]:translate-x-0',
                )}
            />
        </SwitchPrimitive.Root>
    );

    if (standalone) {
        return switchElement;
    }

    const hasLabelContent =
        children !== undefined || onLabel !== undefined || offLabel !== undefined;

    return (
        <label
            data-slot="inline-select-input-toggle"
            className={cn(
                'flex cursor-pointer items-center justify-between gap-3 py-1 select-none',
                disabled && 'cursor-not-allowed opacity-50',
                className,
            )}
        >
            {hasLabelContent ? (
                <span className="text-xs font-medium text-foreground">
                    {children ?? (isChecked ? onLabel : offLabel)}
                </span>
            ) : (
                <span className="font-mono text-xs font-semibold text-foreground">
                    {isChecked ? 'Enabled' : 'Disabled'}
                </span>
            )}
            {switchElement}
        </label>
    );
}

type InlineSelectInputHintProps = React.ComponentProps<'div'>;

function InlineSelectInputHint({ className, ...props }: InlineSelectInputHintProps) {
    return (
        <div
            data-slot="inline-select-input-hint"
            className={cn(
                'mt-1.5 font-mono text-[10.5px] leading-relaxed text-muted-foreground',
                className,
            )}
            {...props}
        />
    );
}

type InlineSelectInputFooterProps = React.ComponentProps<'div'>;

function InlineSelectInputFooter({ className, ...props }: InlineSelectInputFooterProps) {
    return (
        <div
            data-slot="inline-select-input-footer"
            className={cn('mt-2 flex items-center justify-end', className)}
            {...props}
        />
    );
}

type InlineSelectInputDoneProps = React.ComponentProps<typeof PopoverPrimitive.Close>;

function InlineSelectInputDone({
    className,
    children = 'Done',
    ...props
}: InlineSelectInputDoneProps) {
    return (
        <PopoverPrimitive.Close asChild {...props}>
            <button
                type="button"
                data-slot="inline-select-input-done"
                className={cn(
                    'inline-flex cursor-pointer items-center justify-center rounded-xs px-1.5 py-0.5 font-mono text-[10px] font-bold tracking-wider uppercase text-muted-foreground transition-colors',
                    'hover:bg-muted/60 hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none',
                    className,
                )}
            >
                {children}
            </button>
        </PopoverPrimitive.Close>
    );
}

const InlineSelectInput = Object.assign(InlineSelectInputRoot, {
    Trigger: InlineSelectInputTrigger,
    Content: InlineSelectInputContent,
    Label: InlineSelectInputLabel,
    Input: InlineSelectInputInput,
    Range: InlineSelectInputRange,
    Slider: InlineSelectInputRange,
    Toggle: InlineSelectInputToggle,
    Toggler: InlineSelectInputToggle,
    Switch: InlineSelectInputToggle,
    Hint: InlineSelectInputHint,
    Footer: InlineSelectInputFooter,
    Done: InlineSelectInputDone,
    Close: InlineSelectInputDone,
});

export {
    InlineSelectInput,
    type InlineSelectInputProps,
    type InlineSelectInputValue,
    type InlineSelectInputTriggerProps,
    type InlineSelectInputContentProps,
    type InlineSelectInputLabelProps,
    type InlineSelectInputInputProps,
    type InlineSelectInputRangeProps,
    type InlineSelectInputToggleProps,
    type InlineSelectInputHintProps,
    type InlineSelectInputFooterProps,
    type InlineSelectInputDoneProps,
};
