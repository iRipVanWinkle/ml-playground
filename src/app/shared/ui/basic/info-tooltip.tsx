import { type ReactNode } from 'react';
import { InfoIcon } from 'lucide-react';
import { HoverCard } from './hover-card';

export const InfoTooltip = ({ children }: { children: ReactNode }) => {
    return (
        <HoverCard>
            <HoverCard.Trigger asChild>
                <InfoIcon className="h-3 w-3 text-muted-foreground hover:text-primary transition-colors inline" />
            </HoverCard.Trigger>
            <HoverCard.Content className="p-3 text-left text-xs text-muted-foreground">
                <p>{children}</p>
            </HoverCard.Content>
        </HoverCard>
    );
};
