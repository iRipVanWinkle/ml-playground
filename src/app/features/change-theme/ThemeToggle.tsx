import { Moon, Sun, SunMoon } from 'lucide-react';
import { useTheme } from 'next-themes';
import { AVAILABLE_THEMES_LIST, DEFAULT_THEME } from './constants';
import { Button } from '@/app/shared/ui';

export function ThemeToggle() {
    const { theme, setTheme } = useTheme();

    const cycleTheme = () => {
        const themes = AVAILABLE_THEMES_LIST;
        const currentIndex = themes.indexOf(theme ?? DEFAULT_THEME);
        const nextIndex = (currentIndex + 1) % themes.length;
        setTheme(themes[nextIndex]);
    };

    const getThemeIcon = () => {
        switch (theme) {
            case 'light':
                return <Sun className="size-4" />;
            case 'dark':
                return <Moon className="size-4" />;
            case 'system':
            default:
                return <SunMoon className="size-4" />;
        }
    };

    const getThemeLabel = () => {
        switch (theme) {
            case 'light':
                return 'Switch to dark mode';
            case 'dark':
                return 'Switch to system mode';
            case 'system':
            default:
                return 'Switch to light mode';
        }
    };

    return (
        <Button
            variant="ghost"
            size="icon"
            onClick={cycleTheme}
            aria-label={getThemeLabel()}
            title={getThemeLabel()}
        >
            {getThemeIcon()}
        </Button>
    );
}
