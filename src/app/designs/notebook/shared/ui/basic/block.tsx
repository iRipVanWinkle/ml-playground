import * as React from 'react';
import { Slot } from '@radix-ui/react-slot';

import { cn } from '@/app/shared/ui/utils';

/**
 * A callout that explains a single selected option — a normalization method,
 * a transformation — as a title beside supporting content (a formula, a
 * description, ...).
 *
 * ```tsx
 * <Block>
 *     <Block.Title>Z-score</Block.Title>
 *     <Block.Body>
 *         <p>Centers each feature on its mean...</p>
 *     </Block.Body>
 * </Block>
 * ```
 */

type BlockProps = React.ComponentProps<'div'> & { asChild?: boolean };

function BlockRoot({ className, asChild = false, ...props }: BlockProps) {
    const Comp = asChild ? Slot : 'div';

    return (
        <Comp
            data-slot="block"
            className={cn(
                'my-1.5 grid grid-cols-[auto_1fr] gap-x-4.5 rounded-r-lg border-l-2 border-foreground bg-muted px-3.5 py-3',
                className,
            )}
            {...props}
        />
    );
}

type BlockTitleProps = React.ComponentProps<'div'> & { asChild?: boolean };

function BlockTitle({ className, asChild = false, ...props }: BlockTitleProps) {
    const Comp = asChild ? Slot : 'div';

    return (
        <Comp
            data-slot="block-title"
            className={cn('col-start-1 row-start-1 min-w-[110px] text-xs font-bold tracking-tight text-foreground', className)}
            {...props}
        />
    );
}

type BlockBodyProps = React.ComponentProps<'div'> & { asChild?: boolean };

function BlockBody({ className, asChild = false, ...props }: BlockBodyProps) {
    const Comp = asChild ? Slot : 'div';

    return (
        <Comp
            data-slot="block-body"
            className={cn(
                'col-start-2 row-start-1 min-w-0 text-sm leading-relaxed text-foreground',
                '[&_p]:m-0 [&_p]:text-pretty [&_em]:font-medium [&_em]:text-foreground [&_em]:not-italic',
                className,
            )}
            {...props}
        />
    );
}

const Block = Object.assign(BlockRoot, {
    Title: BlockTitle,
    Body: BlockBody,
});

export { Block, type BlockProps, type BlockTitleProps, type BlockBodyProps };
