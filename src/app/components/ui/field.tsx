import * as React from 'react';
import { Label } from './label';
import { cn } from '@/app/lib/utils';

type FieldProps = React.ComponentProps<'div'> & {
    label: string;
    className?: string;
};

function Field({ label, className, children, ...props }: FieldProps) {
    return (
        <div className={cn('grid gap-2', className)} {...props}>
            <Label>{label}</Label>
            {children}
        </div>
    );
}

export { Field };
