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
        const { paperBg, plotBg, textColor, gridColor, legendBg, axisLineColor } = colors;

        return {
            paper_bgcolor: paperBg,
            plot_bgcolor: plotBg,
            font: {
                color: textColor,
            },
            xaxis: {
                gridcolor: gridColor,
                color: textColor,
                zerolinecolor: axisLineColor,
            },
            yaxis: {
                gridcolor: gridColor,
                color: textColor,
                zerolinecolor: axisLineColor,
            },
            scene: {
                xaxis: {
                    gridcolor: gridColor,
                    color: textColor,
                    backgroundcolor: plotBg,
                    zerolinecolor: axisLineColor,
                },
                yaxis: {
                    gridcolor: gridColor,
                    color: textColor,
                    backgroundcolor: plotBg,
                    zerolinecolor: axisLineColor,
                },
                zaxis: {
                    gridcolor: gridColor,
                    color: textColor,
                    backgroundcolor: plotBg,
                    zerolinecolor: axisLineColor,
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
