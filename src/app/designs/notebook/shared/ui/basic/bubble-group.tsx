import * as React from 'react';
import { Slot } from '@radix-ui/react-slot';

import { cn } from '@/app/shared/ui/utils';

import { BubbleGroupContext } from './bubble-group-context';

/**
 * A row of pill-shaped `Bubble`s that share one selection: `type="single"`
 * (the default) keeps at most one bubble on and, by default, lets clicking
 * the active one clear it back to `null`; pass `allowDeselect={false}` for
 * strict radio-button behavior, where the selected bubble can't be cleared
 * by clicking it again. `type="multiple"` lets any number be on at once.
 *
 * ```tsx
 * <BubbleGroup value={normalize} onValueChange={setNormalize}>
 *     <BubbleGroup.Label>Normalize</BubbleGroup.Label>
 *     <Bubble value="zscore">z-score</Bubble>
 *     <Bubble value="minmax">min-max</Bubble>
 * </BubbleGroup>
 *
 * <BubbleGroup type="multiple" value={transforms} onValueChange={setTransforms}>
 *     <BubbleGroup.Label>Transformations</BubbleGroup.Label>
 *     <Bubble value="poly">
 *         polynomial
 *         {transforms.includes('poly') && (
 *             <Bubble.Counter value={degree} onValueChange={setDegree} min={1} max={12} />
 *         )}
 *     </Bubble>
 * </BubbleGroup>
 * ```
 */

type BubbleGroupSingleProps = {
    type?: 'single';
    value: string | null;
    onValueChange: (value: string | null) => void;
    /** Whether clicking the selected bubble clears it back to `null`. Defaults to `true`; set `false` for radio-button behavior. */
    allowDeselect?: boolean;
};

type BubbleGroupMultipleProps = {
    type: 'multiple';
    value: string[];
    onValueChange: (value: string[]) => void;
    allowDeselect?: boolean;
};

type BubbleGroupProps = Omit<React.ComponentProps<'div'>, 'onChange'> & {
    asChild?: boolean;
} & (BubbleGroupSingleProps | BubbleGroupMultipleProps);

function BubbleGroupRoot({
    className,
    asChild = false,
    type = 'single',
    value,
    onValueChange,
    allowDeselect = true,
    ...props
}: BubbleGroupProps) {
    const Comp = asChild ? Slot : 'div';
    const isMultiple = type === 'multiple';

    const isSelected = React.useCallback(
        (item: string) => (isMultiple ? (value as string[]).includes(item) : value === item),
        [isMultiple, value],
    );

    const toggle = React.useCallback(
        (item: string) => {
            if (isMultiple) {
                const items = value as string[];
                (onValueChange as (value: string[]) => void)(
                    items.includes(item) ? items.filter((v) => v !== item) : [...items, item],
                );
            } else {
                const shouldClear = value === item && allowDeselect;
                (onValueChange as (value: string | null) => void)(shouldClear ? null : item);
            }
        },
        [isMultiple, value, onValueChange, allowDeselect],
    );

    const context = React.useMemo(() => ({ isSelected, toggle }), [isSelected, toggle]);

    return (
        <BubbleGroupContext.Provider value={context}>
            <Comp
                data-slot="bubble-group"
                role="group"
                className={cn('flex flex-wrap items-center gap-2', className)}
                {...props}
            />
        </BubbleGroupContext.Provider>
    );
}

type BubbleGroupLabelProps = React.ComponentProps<'span'> & { asChild?: boolean };

function BubbleGroupLabel({ className, asChild = false, ...props }: BubbleGroupLabelProps) {
    const Comp = asChild ? Slot : 'span';

    return (
        <Comp
            data-slot="bubble-group-label"
            className={cn('text-2xs font-semibold tracking-wide text-muted-foreground uppercase', className)}
            {...props}
        />
    );
}

const BubbleGroup = Object.assign(BubbleGroupRoot, {
    Label: BubbleGroupLabel,
});

export { BubbleGroup, type BubbleGroupProps, type BubbleGroupLabelProps };
