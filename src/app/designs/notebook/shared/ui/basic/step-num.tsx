import type * as React from 'react';
import { Slot } from '@radix-ui/react-slot';

import { cn } from '@/app/shared/ui/utils';

import { useOptionalSectionContext } from './section-context';

type StepNumProps = Omit<React.ComponentProps<'div'>, 'children'> & {
    asChild?: boolean;
    step?: number;
    total?: number;
};

const pad = (value: number) => String(value).padStart(2, '0');

function StepNum({ className, asChild = false, step, total, ...props }: StepNumProps) {
    const Comp = asChild ? Slot : 'div';
    const section = useOptionalSectionContext();

    const currentStep = step ?? section?.step;
    const totalSteps = total ?? section?.total;

    if (currentStep === undefined || totalSteps === undefined) {
        throw new Error(
            '`StepNum` needs `step` and `total`, either as props or on the enclosing `Section`',
        );
    }

    return (
        <Comp
            data-slot="step-num"
            data-step={currentStep}
            data-total={totalSteps}
            aria-label={`Step ${currentStep} of ${totalSteps}`}
            className={cn(
                'font-mono text-xs font-bold tracking-widest whitespace-nowrap text-muted-foreground',
                className,
            )}
            {...props}
        >
            <span aria-hidden>
                {pad(currentStep)} <span className="mx-0.5 opacity-40">/</span> {pad(totalSteps)}
            </span>
        </Comp>
    );
}

export { StepNum, type StepNumProps };
