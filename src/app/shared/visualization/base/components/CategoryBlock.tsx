import type { ReactNode } from 'react';

type CategoryBlockProps = {
    title?: string;
    children: ReactNode;
};

export function CategoryBlock({ title, children }: CategoryBlockProps) {
    return (
        <div className="rounded-lg bg-muted/30 p-4 flex flex-col gap-3">
            {title && <h4 className="text-lg font-semibold text-center">{title}</h4>}
            {children}
        </div>
    );
}
