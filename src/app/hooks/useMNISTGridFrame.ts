import { useMemo } from 'react';

export function useMNISTGridFrame(data: number[][]): number[][][] {
    return useMemo(
        () =>
            data.map((digitPixels) => {
                const imageSize = Math.sqrt(digitPixels.length);
                const frame = Array.from({ length: imageSize }, (_, row) =>
                    digitPixels.slice(row * imageSize, (row + 1) * imageSize),
                ).reverse(); // Reverse the rows to correct the orientation

                return frame;
            }),
        [data],
    );
}
