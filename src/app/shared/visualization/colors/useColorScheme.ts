import { useTheme } from 'next-themes';
import {
    DEFAULT_THEME,
    getColorScheme,
    getPlotlyColors,
    NAME_COLORS,
    type ColorScheme,
    type ColorVariant,
    type NameColor,
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
    getColor: (index: number | NameColor, variant?: ColorVariant) => string;
};

export function useColor(): UseColorReturn {
    const scheme = useColorScheme();

    return useMemo(
        () => ({
            getColor: (index: number | NameColor, variant: ColorVariant = 'base') =>
                scheme[variant][
                    typeof index === 'number' ? index % scheme[variant].length : NAME_COLORS[index]
                ],
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
