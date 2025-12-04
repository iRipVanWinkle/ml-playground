export type ThemeType = 'light' | 'dark';
export type ColorVariant = 'base' | 'lighten' | 'darken' | 'fade';
export type ColorPalette = readonly string[];
export type ColorScheme = { [key in ColorVariant]: ColorPalette };

export const DEFAULT_THEME = 'light';

const LIGHT_THEME_COLORS = {
    base: [
        '#0173b2', // Blue - primary
        '#de8f05', // Orange
        '#029e73', // Green
        '#cc3311', // Red
        '#949494', // Gray
        '#eca809', // Yellow
        '#56b4e9', // Light Blue
        '#d55e00', // Vermillion
    ],
    lighten: [
        '#4d9cd4', // Lightened Blue
        '#e8ac4f', // Lightened Orange
        '#4cb798', // Lightened Green
        '#db6a50', // Lightened Red
        '#b4b4b4', // Lightened Gray
        '#f1be49', // Lightened Yellow
        '#8ac9ef', // Lightened Light Blue
        '#e28640', // Lightened Vermillion
    ],
    darken: [
        '#015087', // Darkened Blue
        '#9c6403', // Darkened Orange
        '#016e50', // Darkened Green
        '#8e230b', // Darkened Red
        '#686868', // Darkened Gray
        '#a57606', // Darkened Yellow
        '#3c7ea3', // Darkened Light Blue
        '#953e00', // Darkened Vermillion
    ],
    fade: [
        'rgba(1, 115, 178, 0.4)', // Faded Blue
        'rgba(222, 143, 5, 0.4)', // Faded Orange
        'rgba(2, 158, 115, 0.4)', // Faded Green
        'rgba(204, 51, 17, 0.4)', // Faded Red
        'rgba(148, 148, 148, 0.4)', // Faded Gray
        'rgba(236, 168, 9, 0.4)', // Faded Yellow
        'rgba(86, 180, 233, 0.4)', // Faded Light Blue
        'rgba(213, 94, 0, 0.4)', // Faded Vermillion
    ],
    plotly: {
        paperBg: '#ffffff', // White
        plotBg: 'transparent', // White
        textColor: '#0a0a0a', // Dark Gray
        gridColor: '#e5e5e5', // Light Gray
        legendBg: 'transparent',
    },
} as const;

const DARK_THEME_COLORS = {
    base: [
        '#4a9ef1', // Brighter Blue
        '#fca55d', // Brighter Orange
        '#3fcf8e', // Brighter Green
        '#f05252', // Brighter Red
        '#b8b8b8', // Lighter Gray
        '#ffd166', // Brighter Yellow
        '#7ec8f0', // Brighter Light Blue
        '#ff8a5b', // Brighter Vermillion
    ],
    lighten: [
        '#7cb8f5', // Lightened Blue
        '#fdbf8c', // Lightened Orange
        '#72d9ad', // Lightened Green
        '#f6817e', // Lightened Red
        '#d0d0d0', // Lightened Gray
        '#ffe099', // Lightened Yellow
        '#9fd6f5', // Lightened Light Blue
        '#ffa989', // Lightened Vermillion
    ],
    darken: [
        '#336fa9', // Darkened Blue
        '#b07341', // Darkened Orange
        '#2c9164', // Darkened Green
        '#a83a3a', // Darkened Red
        '#818181', // Darkened Gray
        '#b39247', // Darkened Yellow
        '#588ca8', // Darkened Light Blue
        '#b26140', // Darkened Vermillion
    ],
    fade: [
        'rgba(74, 158, 241, 0.4)', // Faded Blue
        'rgba(252, 165, 93, 0.4)', // Faded Orange
        'rgba(63, 207, 142, 0.4)', // Faded Green
        'rgba(240, 82, 82, 0.4)', // Faded Red
        'rgba(184, 184, 184, 0.4)', // Faded Gray
        'rgba(255, 209, 102, 0.4)', // Faded Yellow
        'rgba(126, 200, 240, 0.4)', // Faded Light Blue
        'rgba(255, 138, 91, 0.4)', // Faded Vermillion
    ],
    plotly: {
        paperBg: '#171717', // Dark Gray
        plotBg: 'transparent', // Dark Gray
        textColor: '#fafafa', // Light Gray
        gridColor: 'rgba(255, 255, 255, 0.1)', // Very Light Gray
        legendBg: 'transparent', // Dark Gray
    },
} as const;

/**
 * Get the color scheme for the theme
 */
export function getColorScheme(theme: ThemeType): ColorScheme {
    return theme === 'dark' ? DARK_THEME_COLORS : LIGHT_THEME_COLORS;
}

/**
 * Get the color for the theme
 */
export function getColor(theme: ThemeType, index: number, variant: ColorVariant = 'base'): string {
    const palette = getColorScheme(theme)[variant];
    return palette[index % palette.length];
}

/**
 * Get the plotly colors for the theme
 */
export function getPlotlyColors(theme: ThemeType) {
    return theme === 'dark' ? DARK_THEME_COLORS.plotly : LIGHT_THEME_COLORS.plotly;
}
