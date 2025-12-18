import * as React from 'react';
import { useRef, useState, useEffect } from 'react';
import * as TabsPrimitive from '@radix-ui/react-tabs';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import { cva, type VariantProps } from 'class-variance-authority';
import { cn } from '../utils';
import { Button } from './button';

const tabsListVariants = cva('inline-flex w-fit items-center', {
    variants: {
        variant: {
            default: 'bg-muted text-muted-foreground h-9 rounded-lg p-[3px]',
            pills: 'gap-1 p-1',
            underline: 'relative gap-1',
        },
        scrollable: {
            false: '',
            true: 'overflow-x-auto scroll-smooth scrollbar-hide [&>*]:flex-shrink-0',
        },
    },
    defaultVariants: {
        variant: 'default',
        scrollable: false,
    },
});

type TabsVariantType = VariantProps<typeof tabsListVariants>;

const TabsVariantContext = React.createContext<TabsVariantType>({});

function EnhancedTabsRoot({
    className,
    variant = 'default',
    scrollable = false,
    ...props
}: React.ComponentProps<typeof TabsPrimitive.Root> & VariantProps<typeof tabsListVariants>) {
    return (
        <TabsVariantContext.Provider value={{ variant, scrollable }}>
            <TabsPrimitive.Root
                data-slot="tabs"
                className={cn('flex flex-col gap-2', scrollable && 'overflow-hidden', className)}
                {...props}
            />
        </TabsVariantContext.Provider>
    );
}

function TabsList({
    className,
    children,
    ...props
}: React.ComponentProps<typeof TabsPrimitive.List>) {
    const scrollContainerRef = useRef<HTMLDivElement>(null);
    const [showLeftArrow, setShowLeftArrow] = useState(false);
    const [showRightArrow, setShowRightArrow] = useState(false);

    const { variant, scrollable } = React.useContext(TabsVariantContext);

    const checkScroll = () => {
        if (scrollContainerRef.current) {
            const { scrollLeft, scrollWidth, clientWidth } = scrollContainerRef.current;
            setShowLeftArrow(scrollLeft > 0);
            setShowRightArrow(scrollLeft < scrollWidth - clientWidth - 1);
        }
    };

    useEffect(() => {
        if (!scrollable) return;

        checkScroll();
        window.addEventListener('resize', checkScroll);
        return () => window.removeEventListener('resize', checkScroll);
    }, [scrollable, children]);

    const scroll = (direction: 'left' | 'right') => {
        if (scrollContainerRef.current) {
            const scrollAmount = 200;
            scrollContainerRef.current.scrollBy({
                left: direction === 'left' ? -scrollAmount : scrollAmount,
                behavior: 'smooth',
            });
        }
    };

    return (
        <div
            className={cn(
                'relative w-full flex items-center',
                variant === 'underline' && 'border-b border-border',
            )}
        >
            {showLeftArrow && (
                <div className="absolute left-0 z-10 h-full flex items-center pr-4 bg-gradient-to-r from-background via-background to-transparent pointer-events-none">
                    <Button
                        onClick={() => scroll('left')}
                        aria-label="Scroll left"
                        size="icon"
                        variant="ghost"
                        className="pointer-events-auto"
                    >
                        <ChevronLeft size={16} />
                    </Button>
                </div>
            )}

            <TabsPrimitive.List
                ref={scrollContainerRef}
                data-slot="tabs-list"
                onScroll={checkScroll}
                className={cn(
                    tabsListVariants({
                        variant,
                        scrollable,
                        className,
                    }),
                )}
                {...props}
            >
                {children}
            </TabsPrimitive.List>

            {showRightArrow && (
                <div className="absolute right-0 z-10 h-full flex items-center pl-4 bg-gradient-to-l from-background via-background to-transparent pointer-events-none">
                    <Button
                        onClick={() => scroll('right')}
                        aria-label="Scroll right"
                        size="icon"
                        variant="ghost"
                        className="pointer-events-auto"
                    >
                        <ChevronRight />
                    </Button>
                </div>
            )}
        </div>
    );
}

const tabsVariants = cva(
    'relative px-4 py-2 font-medium text-sm transition-all duration-300 cursor-pointer inline-flex items-center justify-center gap-2 whitespace-nowrap disabled:pointer-events-none disabled:opacity-50 focus:outline-none',
    {
        variants: {
            variant: {
                default:
                    "data-[state=active]:bg-background dark:data-[state=active]:text-foreground focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:outline-ring dark:data-[state=active]:border-input dark:data-[state=active]:bg-input/30 text-foreground dark:text-muted-foreground inline-flex h-[calc(100%-1px)] flex-1 items-center justify-center gap-1.5 rounded-md border border-transparent px-2 py-1 text-sm font-medium whitespace-nowrap transition-[color,box-shadow] focus-visible:ring-[3px] focus-visible:outline-1 disabled:pointer-events-none disabled:opacity-50 data-[state=active]:shadow-sm [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-4",
                pills: 'rounded-full border data-[state=active]:bg-primary data-[state=active]:text-primary-foreground data-[state=active]:border-primary data-[state=active]:shadow-md data-[state=active]:scale-105 data-[state=inactive]:bg-card data-[state=inactive]:text-muted-foreground data-[state=inactive]:border-border hover:bg-accent hover:text-foreground hover:border-primary/30 hover:shadow-sm',
                underline:
                    'border-b-2 border-t-2 border-t-transparent rounded-none data-[state=active]:text-primary data-[state=active]:border-b-primary data-[state=active]:bg-accent/30 data-[state=inactive]:text-muted-foreground data-[state=inactive]:border-b-transparent data-[state=inactive]:hover:text-foreground data-[state=inactive]:hover:border-b-primary/50 data-[state=inactive]:hover:bg-accent/10',
            },
        },
        defaultVariants: {
            variant: 'default',
        },
    },
);

type TabsTriggerProps = React.ComponentProps<typeof TabsPrimitive.Trigger> & {
    icon?: React.ReactNode;
    badge?: string | number;
};

function TabsTrigger({ className, icon, badge, children, ...props }: TabsTriggerProps) {
    const { variant } = React.useContext(TabsVariantContext);

    return (
        <TabsPrimitive.Trigger
            data-slot="tabs-trigger"
            className={cn(tabsVariants({ variant, className }))}
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

const EnhancedTabs = Object.assign(EnhancedTabsRoot, {
    List: TabsList,
    Trigger: TabsTrigger,
    Content: TabsContent,
});

export { EnhancedTabs };
