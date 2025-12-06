import * as React from 'react';
import { Label } from './label';
import { cn } from '../utils';
import { InfoTooltip } from './info-tooltip';

type FieldProps = React.ComponentProps<'div'> & {
    label: string;
    htmlFor?: string;
    className?: string;
    info?: React.ReactNode;
};

function Field({ label, className, children, htmlFor, info, ...props }: FieldProps) {
    let infoTooltip = null;

    if (info) {
        infoTooltip = <InfoTooltip>{info}</InfoTooltip>;
    }

    return (
        <div className={cn('grid gap-2', className)} {...props}>
            <div className="flex items-center gap-1.5">
                <Label htmlFor={htmlFor}>{label}</Label>
                {infoTooltip}
            </div>
            {children}
        </div>
    );
}

export { Field };
