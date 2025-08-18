import { type ReactNode } from 'react';
import { InfoIcon } from 'lucide-react';
import { HoverCard, HoverCardContent, HoverCardTrigger } from './hover-card';

export const InfoTooltip = ({ children }: { children: ReactNode }) => {
    return (
        <HoverCard>
            <HoverCardTrigger asChild>
                <InfoIcon className="h-3 w-3 text-muted-foreground hover:text-primary transition-colors inline" />
            </HoverCardTrigger>
            <HoverCardContent className="p-3 text-left text-xs text-muted-foreground">
                <p>{children}</p>
            </HoverCardContent>
        </HoverCard>
    );
};
