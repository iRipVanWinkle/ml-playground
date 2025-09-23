import { useId, type ComponentProps, type ElementType, type ReactNode } from 'react';
import { ChevronDown } from 'lucide-react';
import { type VariantProps } from 'class-variance-authority';

import { Button, buttonVariants } from './button';
import { cn } from '@/app/lib/utils';
import {
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuItem,
    DropdownMenuTrigger,
} from './dropdown-menu';

export type MenuItemsType = {
    key?: string | number;
    label: ReactNode;
    onSelect?: (event?: Event) => void;
    disabled?: boolean;
};

type SplitButtonProps<T extends ElementType> = {
    className?: string;
    variant?: VariantProps<typeof buttonVariants>['variant'];
    size?: VariantProps<typeof buttonVariants>['size'];
    children?: ReactNode;
    menuItems?: Array<MenuItemsType>;
    align?: 'start' | 'end' | 'center';
    disabled?: boolean;
} & ComponentProps<T>;

function SplitButton<T extends ElementType = 'button'>({
    className,
    variant,
    children,
    menuItems,
    align = 'end',
    disabled = false,
    ...props
}: SplitButtonProps<T>) {
    const menuId = useId();

    return (
        <DropdownMenu>
            <div
                data-slot="split-button"
                className="inline-flex items-stretch rounded-md shadow-xs"
            >
                <Button
                    data-slot="split-button-primary"
                    variant={variant}
                    className={cn('rounded-r-none', className)}
                    disabled={disabled}
                    {...props}
                >
                    {children}
                </Button>

                <DropdownMenuTrigger asChild>
                    <Button
                        data-slot="split-button-trigger"
                        variant={variant}
                        size="icon"
                        disabled={disabled}
                        className="rounded-l-none px-2"
                        aria-haspopup="menu"
                        aria-controls={menuId}
                    >
                        <ChevronDown className="size-4" aria-hidden />
                    </Button>
                </DropdownMenuTrigger>

                <DropdownMenuContent
                    id={menuId}
                    align={align}
                    sideOffset={6}
                    className="z-50 min-w-[8rem] rounded-md bg-background shadow-md p-1"
                >
                    {menuItems?.map((it: MenuItemsType, idx: number) => (
                        <DropdownMenuItem
                            key={it.key ?? idx}
                            className={cn(
                                'cursor-pointer select-none rounded-sm px-3 py-2 text-sm outline-none data-[disabled]:opacity-50',
                            )}
                            onSelect={it.onSelect}
                            disabled={it.disabled}
                        >
                            {it.label}
                        </DropdownMenuItem>
                    ))}
                </DropdownMenuContent>
            </div>
        </DropdownMenu>
    );
}

export { SplitButton };
