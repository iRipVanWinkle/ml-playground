import * as React from 'react';
import { Slot } from '@radix-ui/react-slot';
import { MinusIcon, PlusIcon } from 'lucide-react';

import { cn } from '@/app/shared/ui/utils';

import { useOptionalBubbleGroupContext } from './bubble-group-context';

/**
 * A pill-shaped, selectable chip. Used standalone as an on/off toggle
 * (`selected` / `onSelectedChange`), or placed inside a `BubbleGroup` to
 * share that group's single- or multi-select state.
 *
 * `Bubble.Counter` is an inline −/value/+ stepper — e.g. a polynomial
 * degree — that a caller drops in as a child, typically only while the
 * bubble is selected.
 *
 * ```tsx
 * <Bubble value="poly">
 *     polynomial
 *     {selected && (
 *         <Bubble.Counter value={degree} onValueChange={setDegree} min={1} max={12} />
 *     )}
 * </Bubble>
 * ```
 */

type BubbleProps = Omit<React.ComponentProps<'button'>, 'value'> & {
    asChild?: boolean;
    /** Identifies this bubble within an enclosing `BubbleGroup`. */
    value: string;
    /** Selected state when used standalone, outside a `BubbleGroup`. */
    selected?: boolean;
    onSelectedChange?: (selected: boolean) => void;
};

function BubbleRoot({
    className,
    asChild = false,
    value,
    selected: selectedProp,
    onSelectedChange,
    children,
    onClick,
    ...props
}: BubbleProps) {
    const Comp = asChild ? Slot : 'button';
    const group = useOptionalBubbleGroupContext();

    const selected = group ? group.isSelected(value) : (selectedProp ?? false);

    function handleClick(event: React.MouseEvent<HTMLButtonElement>) {
        onClick?.(event);
        if (group) group.toggle(value);
        else onSelectedChange?.(!selected);
    }

    return (
        <Comp
            data-slot="bubble"
            type={asChild ? undefined : 'button'}
            data-state={selected ? 'on' : 'off'}
            aria-pressed={selected}
            onClick={handleClick}
            className={cn(
                'inline-flex items-center gap-1 rounded-full border border-input bg-transparent px-3.5 py-2',
                'text-sm font-medium whitespace-nowrap text-foreground transition-colors',
                'hover:bg-accent',
                'focus-visible:ring-[3px] focus-visible:ring-ring/50 focus-visible:outline-none',
                'disabled:cursor-not-allowed disabled:opacity-50',
                'data-[state=on]:border-primary data-[state=on]:bg-primary data-[state=on]:text-primary-foreground data-[state=on]:hover:bg-primary',
                className,
            )}
            {...props}
        >
            {children}
        </Comp>
    );
}

type BubbleCounterProps = {
    value: number;
    onValueChange: (value: number) => void;
    min?: number;
    max?: number;
};

function BubbleCounter({ value, onValueChange, min = -Infinity, max = Infinity }: BubbleCounterProps) {
    return (
        <span
            data-slot="bubble-counter"
            onClick={(event) => event.stopPropagation()}
            className="-mr-1.5 ml-0.5 flex items-center gap-0.5"
        >
            <button
                type="button"
                aria-label="Decrease"
                disabled={value <= min}
                onClick={() => onValueChange(Math.max(min, value - 1))}
                className="inline-grid size-5 place-items-center rounded-full transition-colors hover:bg-primary-foreground/15 disabled:pointer-events-none disabled:opacity-40"
            >
                <MinusIcon className="size-3" />
            </button>
            <span className="w-5 text-center font-mono text-xs font-bold tabular-nums">{value}</span>
            <button
                type="button"
                aria-label="Increase"
                disabled={value >= max}
                onClick={() => onValueChange(Math.min(max, value + 1))}
                className="inline-grid size-5 place-items-center rounded-full transition-colors hover:bg-primary-foreground/15 disabled:pointer-events-none disabled:opacity-40"
            >
                <PlusIcon className="size-3" />
            </button>
        </span>
    );
}

const Bubble = Object.assign(BubbleRoot, {
    Counter: BubbleCounter,
});

export { Bubble, type BubbleProps, type BubbleCounterProps };
