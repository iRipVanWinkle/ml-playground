import type { ReactNode } from 'react';
import { ThemeProvider as NextThemeProvider } from 'next-themes';
import { DEFAULT_THEME } from './constants';

export function ThemeProvider({ children }: { children: ReactNode }) {
    return (
        <NextThemeProvider
            attribute="class"
            defaultTheme={DEFAULT_THEME}
            enableSystem
            enableColorScheme
            disableTransitionOnChange
        >
            {children}
        </NextThemeProvider>
    );
}
