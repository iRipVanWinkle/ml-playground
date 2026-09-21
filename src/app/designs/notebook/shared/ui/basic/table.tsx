import * as React from 'react';
import { Slot } from '@radix-ui/react-slot';

import { cn } from '@/app/shared/ui/utils';

/**
 * Data preview table modeled after MI (Remix) variation-b.jsx and shadcn/ui Table.
 *
 * Supports compound component composition (`Table.Header`, `Table.Row`, `Table.Cell`,
 * `Table.Wrap`, `Table.Meta`, `Table.Scroll`, `Table.Caption`, `Table.Description`)
 * as well as individual named exports.
 *
 * ```tsx
 * <Table.Wrap>
 *     <Table.Meta>
 *         <span>· Preview · first 4 of <b>150</b></span>
 *         <span>target column · <b>species</b></span>
 *     </Table.Meta>
 *     <Table>
 *         <Table.Header>
 *             <Table.Row>
 *                 <Table.Head>sepal_length</Table.Head>
 *                 <Table.Head>species</Table.Head>
 *             </Table.Row>
 *         </Table.Header>
 *         <Table.Body>
 *             <Table.Row>
 *                 <Table.Cell>5.1</Table.Cell>
 *                 <Table.Cell isLabel>setosa</Table.Cell>
 *             </Table.Row>
 *             <Table.Row>
 *                 <Table.Cell colSpan={2} muted>
 *                     … 146 more
 *                 </Table.Cell>
 *             </Table.Row>
 *         </Table.Body>
 *     </Table>
 *     <Table.Description>
 *         Data is shuffled with seed 42
 *     </Table.Description>
 * </Table.Wrap>
 * ```
 */

type TableWrapProps = React.ComponentProps<'div'> & { asChild?: boolean };

function TableWrap({ className, asChild = false, ...props }: TableWrapProps) {
    const Comp = asChild ? Slot : 'div';

    return <Comp data-slot="table-wrap" className={cn('my-6', className)} {...props} />;
}

type TableMetaProps = React.ComponentProps<'div'> & { asChild?: boolean };

function TableMeta({ className, asChild = false, ...props }: TableMetaProps) {
    const Comp = asChild ? Slot : 'div';

    return (
        <Comp
            data-slot="table-meta"
            className={cn(
                'mb-2 flex flex-wrap items-baseline justify-between gap-3 font-mono text-[10.5px] font-bold tracking-[1.3px] uppercase text-muted-foreground',
                '[&_b]:font-bold [&_b]:text-foreground',
                className,
            )}
            {...props}
        />
    );
}

type TableScrollProps = React.ComponentProps<'div'> & { asChild?: boolean };

function TableScroll({ className, asChild = false, ...props }: TableScrollProps) {
    const Comp = asChild ? Slot : 'div';

    return (
        <Comp
            data-slot="table-scroll"
            className={cn('relative my-3 w-full overflow-x-auto', className)}
            {...props}
        />
    );
}

type TableProps = React.ComponentProps<'table'> & {
    asChild?: boolean;
    containerClassName?: string;
    noScroll?: boolean;
};

function TableRoot({
    className,
    asChild = false,
    containerClassName,
    noScroll = false,
    ...props
}: TableProps) {
    if (asChild) {
        return (
            <Slot
                data-slot="table"
                className={cn(
                    'w-full border-collapse font-mono text-xs text-left tabular-nums caption-bottom',
                    className,
                )}
                {...props}
            />
        );
    }

    const tableElement = (
        <table
            data-slot="table"
            className={cn(
                'w-full border-collapse font-mono text-xs text-left tabular-nums caption-bottom',
                className,
            )}
            {...props}
        />
    );

    if (noScroll) {
        return tableElement;
    }

    return (
        <div
            data-slot="table-scroll"
            className={cn('relative my-3 w-full overflow-x-auto', containerClassName)}
        >
            {tableElement}
        </div>
    );
}

type TableHeaderProps = React.ComponentProps<'thead'> & { asChild?: boolean };

function TableHeader({ className, asChild = false, ...props }: TableHeaderProps) {
    const Comp = asChild ? Slot : 'thead';

    return (
        <Comp
            data-slot="table-header"
            className={cn('[&_tr]:border-b [&_tr]:border-border', className)}
            {...props}
        />
    );
}

type TableBodyProps = React.ComponentProps<'tbody'> & { asChild?: boolean };

function TableBody({ className, asChild = false, ...props }: TableBodyProps) {
    const Comp = asChild ? Slot : 'tbody';

    return (
        <Comp
            data-slot="table-body"
            className={cn('[&_tr:last-child]:border-0', className)}
            {...props}
        />
    );
}

type TableFooterProps = React.ComponentProps<'tfoot'> & { asChild?: boolean };

function TableFooter({ className, asChild = false, ...props }: TableFooterProps) {
    const Comp = asChild ? Slot : 'tfoot';

    return (
        <Comp
            data-slot="table-footer"
            className={cn(
                'bg-transparent font-normal [&_td]:border-b-0 [&>tr]:last:border-b-0',
                className,
            )}
            {...props}
        />
    );
}

type TableRowProps = React.ComponentProps<'tr'> & { asChild?: boolean };

function TableRow({ className, asChild = false, ...props }: TableRowProps) {
    const Comp = asChild ? Slot : 'tr';

    return (
        <Comp
            data-slot="table-row"
            className={cn(
                'border-b border-dashed border-border transition-colors hover:bg-muted/50 data-[state=selected]:bg-muted',
                className,
            )}
            {...props}
        />
    );
}

type TableHeadProps = React.ComponentProps<'th'> & { asChild?: boolean };

function TableHead({ className, asChild = false, ...props }: TableHeadProps) {
    const Comp = asChild ? Slot : 'th';

    return (
        <Comp
            data-slot="table-head"
            className={cn(
                'h-9 px-2.5 py-2 text-left align-middle font-mono text-[11px] font-semibold tracking-[0.5px] uppercase text-muted-foreground whitespace-nowrap border-b border-border',
                className,
            )}
            {...props}
        />
    );
}

type TableCellProps = React.ComponentProps<'td'> & {
    asChild?: boolean;
    isLabel?: boolean;
    muted?: boolean;
};

function TableCell({
    className,
    asChild = false,
    isLabel = false,
    muted = false,
    ...props
}: TableCellProps) {
    const Comp = asChild ? Slot : 'td';

    return (
        <Comp
            data-slot="table-cell"
            className={cn(
                'px-2.5 py-1.5 align-middle font-mono text-xs whitespace-nowrap border-b border-dashed border-border text-foreground',
                isLabel && 'font-semibold text-blue-600 dark:text-blue-400',
                muted && 'text-muted-foreground',
                className,
            )}
            {...props}
        />
    );
}

type TableCaptionProps = React.ComponentProps<'caption'> & { asChild?: boolean };

function TableCaption({ className, asChild = false, ...props }: TableCaptionProps) {
    const Comp = asChild ? Slot : 'caption';

    return (
        <Comp
            data-slot="table-caption"
            className={cn('mt-3 text-xs leading-relaxed text-muted-foreground', className)}
            {...props}
        />
    );
}

type TableDescriptionProps = React.ComponentProps<'p'> & { asChild?: boolean };

function TableDescription({ className, asChild = false, ...props }: TableDescriptionProps) {
    const Comp = asChild ? Slot : 'p';

    return (
        <Comp
            data-slot="table-description"
            className={cn(
                'mt-2.5 text-[13px] leading-[1.55] font-light text-muted-foreground',
                '[&_b]:font-semibold [&_b]:text-foreground',
                className,
            )}
            {...props}
        />
    );
}

const Table = Object.assign(TableRoot, {
    Wrap: TableWrap,
    Meta: TableMeta,
    Scroll: TableScroll,
    Header: TableHeader,
    Body: TableBody,
    Footer: TableFooter,
    Row: TableRow,
    Head: TableHead,
    Cell: TableCell,
    Caption: TableCaption,
    Description: TableDescription,
});

export {
    Table,
    TableWrap,
    TableMeta,
    TableScroll,
    TableHeader,
    TableBody,
    TableFooter,
    TableRow,
    TableHead,
    TableCell,
    TableCaption,
    TableDescription,
    type TableProps,
    type TableWrapProps,
    type TableMetaProps,
    type TableScrollProps,
    type TableHeaderProps,
    type TableBodyProps,
    type TableFooterProps,
    type TableRowProps,
    type TableHeadProps,
    type TableCellProps,
    type TableCaptionProps,
    type TableDescriptionProps,
};
