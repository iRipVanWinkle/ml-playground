import { Button, GitHub } from '@/app/shared/ui';

export function Footer() {
    return (
        <footer className="border-t bg-background p-2">
            <div className="container mx-auto flex justify-center">
                <Button variant="ghost" size="icon" asChild>
                    <a
                        href="https://github.com/iRipVanWinkle/ml-playground"
                        target="_blank"
                        rel="noopener noreferrer"
                        aria-label="View source code on GitHub"
                        title="View source code on GitHub"
                    >
                        <GitHub className="size-8" />
                    </a>
                </Button>
            </div>
        </footer>
    );
}
