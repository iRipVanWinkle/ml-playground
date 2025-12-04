import { useTheme } from 'next-themes';
import {
    DEFAULT_THEME,
    getColorScheme,
    getPlotlyColors,
    type ColorScheme,
    type ColorVariant,
    type ThemeType,
} from './palette';
import { useMemo } from 'react';

/**
 * Get the effective theme based on the system theme and the user theme
 */
function useEffectiveTheme(): ThemeType {
    const { theme, systemTheme } = useTheme();
    return ((theme === 'system' ? systemTheme : theme) ?? DEFAULT_THEME) as ThemeType;
}

/**
 * Get the color scheme for the effective theme
 */
export function useColorScheme(): ColorScheme {
    const theme = useEffectiveTheme();

    return getColorScheme(theme);
}

/**
 * Get the color for the effective theme
 */
type UseColorReturn = {
    getColor: (index: number, variant?: ColorVariant) => string;
};

export function useColor(): UseColorReturn {
    const scheme = useColorScheme();

    return useMemo(
        () => ({
            getColor: (index: number, variant: ColorVariant = 'base') =>
                scheme[variant][index % scheme[variant].length],
        }),
        [scheme],
    );
}

/**
 * Get the plotly colors for the effective theme
 */
type UsePlotlyColorsReturn = ReturnType<typeof getPlotlyColors>;

export function usePlotlyColors(): UsePlotlyColorsReturn {
    const theme = useEffectiveTheme();

    return useMemo(() => getPlotlyColors(theme), [theme]);
}
