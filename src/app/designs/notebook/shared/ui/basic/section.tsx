import * as React from 'react';
import { Slot } from '@radix-ui/react-slot';

import { cn } from '@/app/shared/ui/utils';

import { SectionContext } from './section-context';

type SectionProps = React.ComponentProps<'section'> & {
    asChild?: boolean;
    step?: number;
    total?: number;
};

function SectionRoot({ className, asChild = false, step, total, ...props }: SectionProps) {
    const Comp = asChild ? Slot : 'section';

    const context = { step, total };

    return (
        <SectionContext.Provider value={context}>
            <Comp
                data-slot="section"
                className={cn('mb-14 last-of-type:mb-0 md:mb-18', className)}
                {...props}
            />
        </SectionContext.Provider>
    );
}

type SectionHeaderProps = React.ComponentProps<'header'> & { asChild?: boolean };

function SectionHeader({ className, asChild = false, ...props }: SectionHeaderProps) {
    const Comp = asChild ? Slot : 'header';

    return (
        <Comp
            data-slot="section-header"
            className={cn('mb-4 flex items-baseline gap-3.5', className)}
            {...props}
        />
    );
}

type SectionTitleProps = React.ComponentProps<'h2'> & { asChild?: boolean };

function SectionTitle({ className, asChild = false, ...props }: SectionTitleProps) {
    const Comp = asChild ? Slot : 'h2';

    return (
        <Comp
            data-slot="section-title"
            className={cn(
                'text-2xl leading-tight font-semibold tracking-tight text-balance text-foreground',
                className,
            )}
            {...props}
        />
    );
}

type SectionBodyProps = React.ComponentProps<'div'> & { asChild?: boolean };

function SectionBody({ className, asChild = false, ...props }: SectionBodyProps) {
    const Comp = asChild ? Slot : 'div';

    return (
        <Comp
            data-slot="section-body"
            className={cn(
                'text-sm leading-relaxed text-foreground',
                '[&_p]:mb-4 [&_p]:text-pretty [&_p:last-child]:mb-0',
                className,
            )}
            {...props}
        />
    );
}

type SectionHelperProps = React.ComponentProps<'p'> & { asChild?: boolean };

function SectionHelper({ className, asChild = false, ...props }: SectionHelperProps) {
    const Comp = asChild ? Slot : 'p';

    return (
        <Comp
            data-slot="section-helper"
            className={cn(
                'mt-1 mb-4 max-w-xl text-xs leading-normal text-pretty text-muted-foreground',
                className,
            )}
            {...props}
        />
    );
}

const Section = Object.assign(SectionRoot, {
    Header: SectionHeader,
    Title: SectionTitle,
    Body: SectionBody,
    Helper: SectionHelper,
});

export {
    Section,
    type SectionProps,
    type SectionHeaderProps,
    type SectionTitleProps,
    type SectionBodyProps,
    type SectionHelperProps,
};
