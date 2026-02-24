import { type ReactNode } from 'react';
import { InfoIcon } from 'lucide-react';
import { HoverCard } from './hover-card';

export const InfoTooltip = ({ children }: { children: ReactNode }) => {
    return (
        <HoverCard>
            <HoverCard.Trigger>
                <button
                    type="button"
                    className="inline-flex items-center justify-center focus-visible:ring-ring/50 focus-visible:ring-[3px] rounded-sm outline-none"
                    aria-label="More information"
                >
                    <InfoIcon
                        className="h-3 w-3 text-muted-foreground hover:text-primary transition-colors"
                        aria-hidden="true"
                    />
                </button>
            </HoverCard.Trigger>

            <HoverCard.Content className="p-3 text-left text-xs text-muted-foreground">
                <p>{children}</p>
            </HoverCard.Content>
        </HoverCard>
    );
};
