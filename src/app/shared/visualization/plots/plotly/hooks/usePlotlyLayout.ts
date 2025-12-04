import { useMemo } from 'react';
import type { Layout } from 'plotly.js';
import { usePlotlyColors } from '../../../colors/useColorScheme';

/**
 * Returns a themed Plotly layout configuration based on the current color scheme.
 * Automatically adapts to light/dark theme changes.
 */
export function usePlotlyLayout(): Partial<Layout> {
    const colors = usePlotlyColors();

    return useMemo(() => {
        const { paperBg, plotBg, textColor, gridColor, legendBg } = colors;

        return {
            paper_bgcolor: paperBg,
            plot_bgcolor: plotBg,
            font: {
                color: textColor,
            },
            xaxis: {
                gridcolor: gridColor,
                color: textColor,
            },
            yaxis: {
                gridcolor: gridColor,
                color: textColor,
            },
            scene: {
                xaxis: {
                    gridcolor: gridColor,
                    color: textColor,
                    backgroundcolor: plotBg,
                },
                yaxis: {
                    gridcolor: gridColor,
                    color: textColor,
                    backgroundcolor: plotBg,
                },
                zaxis: {
                    gridcolor: gridColor,
                    color: textColor,
                    backgroundcolor: plotBg,
                },
            },
            legend: {
                bgcolor: legendBg,
                bordercolor: gridColor,
                font: {
                    color: textColor,
                },
            },
        };
    }, [colors]);
}
