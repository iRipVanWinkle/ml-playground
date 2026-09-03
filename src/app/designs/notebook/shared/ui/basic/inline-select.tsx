import type * as React from 'react';
import * as SelectPrimitive from '@radix-ui/react-select';
import { ChevronDownIcon, ChevronUpIcon } from 'lucide-react';

import { cn } from '@/app/shared/ui/utils';

/**
 * A select that lives inside a sentence: the trigger reads as a dashed-underlined
 * word in prose, the menu is a regular listbox popover. Built on Radix Select, so
 * keyboard navigation, typeahead and `listbox`/`option` semantics come for free.
 *
 * ```tsx
 * <p>
 *     I want to{' '}
 *     <InlineSelect value={task} onValueChange={setTask}>
 *         <InlineSelect.Trigger placeholder="pick a task" />
 *         <InlineSelect.Content>
 *             <InlineSelect.Item value="regression" hint="regression">
 *                 predict a number
 *             </InlineSelect.Item>
 *         </InlineSelect.Content>
 *     </InlineSelect>
 *     .
 * </p>
 * ```
 */

type InlineSelectProps = React.ComponentProps<typeof SelectPrimitive.Root>;

function InlineSelectRoot(props: InlineSelectProps) {
    return <SelectPrimitive.Root data-slot="inline-select" {...props} />;
}

type InlineSelectTriggerProps = Omit<
    React.ComponentProps<typeof SelectPrimitive.Trigger>,
    'children'
> & {
    /** Shown while nothing is selected. */
    placeholder?: string;
    /** Renders the selected value; defaults to the item's own text. */
    children?: React.ReactNode;
};

function InlineSelectTrigger({
    className,
    placeholder,
    children,
    ...props
}: InlineSelectTriggerProps) {
    return (
        <SelectPrimitive.Trigger
            data-slot="inline-select-trigger"
            className={cn(
                'inline cursor-pointer rounded-xs border-0 border-b border-dashed p-0 px-0.5 font-[inherit] text-[length:inherit] font-medium text-foreground italic',
                'border-muted-foreground/60 bg-transparent box-decoration-clone transition-colors outline-none',
                'hover:border-foreground data-[state=open]:border-foreground',
                'focus-visible:ring-[3px] focus-visible:ring-ring/50 focus-visible:rounded-sm',
                'data-placeholder:text-muted-foreground',
                'disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:border-muted-foreground/60',
                '[&_svg]:pointer-events-none [&_svg]:inline [&_svg]:size-3.5 [&_svg]:shrink-0 [&_svg]:align-middle [&_svg]:opacity-60 [&_svg]:ml-1',
                '[&_svg]:transition-transform data-[state=open]:[&_svg]:rotate-180',
                className,
            )}
            {...props}
        >
            {children ?? <SelectPrimitive.Value placeholder={placeholder} />}
            <SelectPrimitive.Icon asChild>
                <ChevronDownIcon aria-hidden />
            </SelectPrimitive.Icon>
        </SelectPrimitive.Trigger>
    );
}

type InlineSelectContentProps = React.ComponentProps<typeof SelectPrimitive.Content>;

function InlineSelectContent({
    className,
    children,
    align = 'start',
    sideOffset = 8,
    ...props
}: InlineSelectContentProps) {
    return (
        <SelectPrimitive.Portal>
            <SelectPrimitive.Content
                data-slot="inline-select-content"
                position="popper"
                align={align}
                sideOffset={sideOffset}
                className={cn(
                    'relative z-50 max-h-(--radix-select-content-available-height) min-w-[16rem] origin-(--radix-select-content-transform-origin) overflow-x-hidden overflow-y-auto',
                    'rounded-xl border bg-popover p-1.5 text-sm font-normal tracking-normal text-popover-foreground not-italic shadow-md',
                    'data-[state=open]:animate-in data-[state=open]:fade-in-0 data-[state=open]:zoom-in-95',
                    'data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=closed]:zoom-out-95',
                    'data-[side=bottom]:slide-in-from-top-2 data-[side=top]:slide-in-from-bottom-2 data-[side=left]:slide-in-from-right-2 data-[side=right]:slide-in-from-left-2',
                    className,
                )}
                {...props}
            >
                <InlineSelectScrollUpButton />
                <SelectPrimitive.Viewport className="scroll-my-1">
                    {children}
                </SelectPrimitive.Viewport>
                <InlineSelectScrollDownButton />
            </SelectPrimitive.Content>
        </SelectPrimitive.Portal>
    );
}

type InlineSelectItemProps = React.ComponentProps<typeof SelectPrimitive.Item> & {
    /** Secondary label shown on the right — the technical name behind the plain-language option. */
    hint?: React.ReactNode;
};

function InlineSelectItem({ className, children, hint, ...props }: InlineSelectItemProps) {
    return (
        <SelectPrimitive.Item
            data-slot="inline-select-item"
            className={cn(
                'group/item relative flex w-full cursor-pointer items-baseline justify-between gap-3 rounded-lg px-3 py-2',
                'text-sm font-medium outline-none select-none',
                'focus:bg-accent focus:text-accent-foreground',
                'data-[state=checked]:bg-primary data-[state=checked]:text-primary-foreground',
                'data-disabled:pointer-events-none data-disabled:opacity-50',
                className,
            )}
            {...props}
        >
            <SelectPrimitive.ItemText>{children}</SelectPrimitive.ItemText>
            {hint !== undefined && (
                <span
                    data-slot="inline-select-item-hint"
                    className="font-mono text-[11px] font-normal whitespace-nowrap text-muted-foreground group-data-[state=checked]/item:text-inherit group-data-[state=checked]/item:opacity-70"
                >
                    {hint}
                </span>
            )}
        </SelectPrimitive.Item>
    );
}

type InlineSelectGroupProps = React.ComponentProps<typeof SelectPrimitive.Group>;

function InlineSelectGroup(props: InlineSelectGroupProps) {
    return <SelectPrimitive.Group data-slot="inline-select-group" {...props} />;
}

type InlineSelectLabelProps = React.ComponentProps<typeof SelectPrimitive.Label>;

function InlineSelectLabel({ className, ...props }: InlineSelectLabelProps) {
    return (
        <SelectPrimitive.Label
            data-slot="inline-select-label"
            className={cn(
                'px-3 pt-2.5 pb-1.5 font-mono text-[10px] font-bold tracking-widest uppercase text-muted-foreground',
                className,
            )}
            {...props}
        />
    );
}

type InlineSelectSeparatorProps = React.ComponentProps<typeof SelectPrimitive.Separator>;

function InlineSelectSeparator({ className, ...props }: InlineSelectSeparatorProps) {
    return (
        <SelectPrimitive.Separator
            data-slot="inline-select-separator"
            className={cn('-mx-1.5 my-1.5 h-px bg-border', className)}
            {...props}
        />
    );
}

function InlineSelectScrollUpButton({
    className,
    ...props
}: React.ComponentProps<typeof SelectPrimitive.ScrollUpButton>) {
    return (
        <SelectPrimitive.ScrollUpButton
            data-slot="inline-select-scroll-up-button"
            className={cn('flex cursor-default items-center justify-center py-1', className)}
            {...props}
        >
            <ChevronUpIcon className="size-4 opacity-60" />
        </SelectPrimitive.ScrollUpButton>
    );
}

function InlineSelectScrollDownButton({
    className,
    ...props
}: React.ComponentProps<typeof SelectPrimitive.ScrollDownButton>) {
    return (
        <SelectPrimitive.ScrollDownButton
            data-slot="inline-select-scroll-down-button"
            className={cn('flex cursor-default items-center justify-center py-1', className)}
            {...props}
        >
            <ChevronDownIcon className="size-4 opacity-60" />
        </SelectPrimitive.ScrollDownButton>
    );
}

const InlineSelect = Object.assign(InlineSelectRoot, {
    Trigger: InlineSelectTrigger,
    Content: InlineSelectContent,
    Item: InlineSelectItem,
    Group: InlineSelectGroup,
    Label: InlineSelectLabel,
    Separator: InlineSelectSeparator,
});

export {
    InlineSelect,
    type InlineSelectProps,
    type InlineSelectTriggerProps,
    type InlineSelectContentProps,
    type InlineSelectItemProps,
    type InlineSelectGroupProps,
    type InlineSelectLabelProps,
    type InlineSelectSeparatorProps,
};
