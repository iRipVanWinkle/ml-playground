import { type ReactNode } from 'react';
import { HoverCard, HoverCardContent, HoverCardTrigger } from './hover-card';

type TooltipWrapperProps = {
    tooltip?: ReactNode;
    children: ReactNode;
};

export const TooltipWrapper = ({ tooltip, children }: TooltipWrapperProps) => {
    if (!tooltip) {
        return children;
    }

    return (
        <HoverCard openDelay={250}>
            <HoverCardTrigger asChild>{children}</HoverCardTrigger>
            <HoverCardContent className="p-3 text-left text-xs text-muted-foreground">
                {tooltip}
            </HoverCardContent>
        </HoverCard>
    );
};
