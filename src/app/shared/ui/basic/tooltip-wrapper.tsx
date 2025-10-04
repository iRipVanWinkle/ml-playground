import { type ReactNode } from 'react';
import { HoverCard } from './hover-card';

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
            <HoverCard.Trigger asChild>{children}</HoverCard.Trigger>
            <HoverCard.Content className="p-3 text-left text-xs text-muted-foreground">
                {tooltip}
            </HoverCard.Content>
        </HoverCard>
    );
};
