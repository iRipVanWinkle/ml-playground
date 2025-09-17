import { Button } from './ui/button';
import Github from './icons/GitHub';

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
                        <Github className="size-8" />
                    </a>
                </Button>
            </div>
        </footer>
    );
}
