import * as React from 'react';
import * as TabsPrimitive from '@radix-ui/react-tabs';

import { cn } from '@/app/lib/utils';

const TabsVariantContext = React.createContext<'default' | 'pills' | 'underline'>('default');

function Tabs({ className, ...props }: React.ComponentProps<typeof TabsPrimitive.Root>) {
    return (
        <TabsPrimitive.Root
            data-slot="tabs"
            className={cn('flex flex-col gap-2', className)}
            {...props}
        />
    );
}

interface TabsListProps extends React.ComponentProps<typeof TabsPrimitive.List> {
    variant?: 'default' | 'pills' | 'underline';
}

function TabsList({ className, variant = 'default', ...props }: TabsListProps) {
    const getListVariantClasses = (variant: TabsListProps['variant']) => {
        const baseClasses = 'inline-flex w-fit items-center';

        switch (variant) {
            case 'pills':
                return cn(baseClasses, 'gap-1 p-1');
            case 'underline':
                return cn(baseClasses, 'relative gap-1');
            default:
                return cn(
                    baseClasses,
                    'bg-muted text-muted-foreground h-9 rounded-lg p-[3px] gap-1',
                );
        }
    };

    return (
        <TabsVariantContext.Provider value={variant}>
            <TabsPrimitive.List
                data-slot="tabs-list"
                className={cn(getListVariantClasses(variant), className)}
                {...props}
            />
        </TabsVariantContext.Provider>
    );
}

interface TabsTriggerProps extends React.ComponentProps<typeof TabsPrimitive.Trigger> {
    icon?: React.ReactNode;
    badge?: string | number;
}

const getTabVariantClasses = (variant: 'default' | 'pills' | 'underline') => {
    const baseClasses =
        'relative px-4 py-2 font-medium text-sm transition-all duration-300 cursor-pointer inline-flex items-center justify-center gap-2 whitespace-nowrap disabled:pointer-events-none disabled:opacity-50 focus:outline-none';

    switch (variant) {
        case 'pills':
            return cn(
                baseClasses,
                'rounded-full border',
                'data-[state=active]:bg-primary data-[state=active]:text-primary-foreground data-[state=active]:border-primary data-[state=active]:shadow-md data-[state=active]:scale-105',
                'data-[state=inactive]:bg-card data-[state=inactive]:text-muted-foreground data-[state=inactive]:border-border',
                'hover:bg-accent hover:text-foreground hover:border-primary/30 hover:shadow-sm',
            );

        case 'underline':
            return cn(
                baseClasses,
                'border-b-2 border-t-2 border-t-transparent rounded-none',
                'data-[state=active]:text-primary data-[state=active]:border-b-primary data-[state=active]:bg-accent/30',
                'data-[state=inactive]:text-muted-foreground data-[state=inactive]:border-b-transparent',
                'data-[state=inactive]:hover:text-foreground data-[state=inactive]:hover:border-b-primary/50 data-[state=inactive]:hover:bg-accent/10',
            );

        default:
            return cn(
                baseClasses,
                'h-[calc(100%-1px)] flex-1 rounded-md border border-transparent',
                'data-[state=active]:bg-background data-[state=active]:text-foreground data-[state=active]:shadow-sm',
                'data-[state=inactive]:text-muted-foreground',
                'hover:bg-accent hover:text-foreground',
                'focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:outline-ring focus-visible:ring-[3px] focus-visible:outline-1',
            );
    }
};

function TabsTrigger({ className, icon, badge, children, ...props }: TabsTriggerProps) {
    const variant = React.useContext(TabsVariantContext);

    return (
        <TabsPrimitive.Trigger
            data-slot="tabs-trigger"
            className={cn(getTabVariantClasses(variant), className)}
            {...props}
        >
            {icon && <span className="flex-shrink-0 w-4 h-4 [&>svg]:w-4 [&>svg]:h-4">{icon}</span>}
            <span>{children}</span>
            {badge && (
                <span
                    className={cn(
                        'ml-1 px-2 py-0.5 text-xs rounded-full font-semibold',
                        'data-[state=active]:bg-primary-foreground/20 data-[state=active]:text-primary-foreground',
                        'data-[state=inactive]:bg-primary/10 data-[state=inactive]:text-primary',
                    )}
                >
                    {badge}
                </span>
            )}
        </TabsPrimitive.Trigger>
    );
}

function TabsContent({ className, ...props }: React.ComponentProps<typeof TabsPrimitive.Content>) {
    return (
        <TabsPrimitive.Content
            data-slot="tabs-content"
            className={cn('flex-1 outline-none', className)}
            {...props}
        />
    );
}

export { Tabs, TabsList, TabsTrigger, TabsContent };
export type { TabsListProps, TabsTriggerProps };
