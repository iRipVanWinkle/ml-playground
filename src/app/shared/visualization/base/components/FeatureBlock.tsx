import type { ReactNode } from 'react';

type FeatureBlockProps = {
    title: string;
    children: ReactNode;
};

export function FeatureBlock({ title, children }: FeatureBlockProps) {
    return (
        <div className="flex flex-col gap-3">
            <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground text-center py-2">
                {title}
            </div>
            {children}
        </div>
    );
}
