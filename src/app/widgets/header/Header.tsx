import { ThemeToggle } from '@/app/features/change-theme';
import { ML } from '@/app/shared/ui';

export function Header() {
    return (
        <header className="flex items-center justify-between w-full p-2 border-b bg-background">
            <div
                className="flex items-center justify-between container mx-auto"
                style={{ maxWidth: '1280px', margin: '0 auto' }}
            >
                <div className="flex items-center gap-2">
                    <ML className="size-6" />
                    <h1 className="text-lg font-semibold text-foreground">ML Playground</h1>
                </div>

                <ThemeToggle />
            </div>
        </header>
    );
}
