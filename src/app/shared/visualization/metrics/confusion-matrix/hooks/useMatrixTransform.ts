interface UseMatrixTransformParams {
    size: number;
}

interface UseMatrixTransformReturn {
    gridCallbackRef: (element: HTMLDivElement) => void;
}

/**
 * Hook to calculate and apply transform offset for centering the matrix grid
 */
export function useMatrixTransform({ size }: UseMatrixTransformParams): UseMatrixTransformReturn {
    const gridCallbackRef = (element: HTMLDivElement) => {
        const grid = element as HTMLDivElement;
        const container = element.parentElement as HTMLDivElement;

        const updateTransformation = () => {
            if (!container || !grid) return;

            const currentTransform = grid.style.transform;
            grid.style.transform = 'none';

            const containerRect = container.getBoundingClientRect();
            if (containerRect.width === 0) {
                grid.style.transform = currentTransform;
                return;
            }
            const containerCenterX = containerRect.width / 2;

            const gridRect = grid.getBoundingClientRect();

            const gridItems = grid.querySelectorAll('[data-type="matrix-cell"]');
            const firstCellElement: HTMLElement = gridItems[0] as HTMLElement;
            const lastCellElement: HTMLElement = gridItems[size - 1] as HTMLElement;

            if (!firstCellElement || !lastCellElement) {
                grid.style.transform = currentTransform;
                return;
            }

            const firstCellRect = firstCellElement.getBoundingClientRect();
            const lastCellRect = lastCellElement.getBoundingClientRect();

            const coloredGridLeft = firstCellRect.left - gridRect.left;
            const coloredGridRight = lastCellRect.right - gridRect.left;

            const coloredGridCenterX = (coloredGridLeft + coloredGridRight) / 2;

            const gridOffsetX = gridRect.left - containerRect.left;

            const coloredGridCenterXInContainer = gridOffsetX + coloredGridCenterX;

            const offsetX = containerCenterX - coloredGridCenterXInContainer;

            const gridWidth = gridRect.width;
            const minOffsetX = -gridOffsetX;
            const maxOffsetX = containerRect.width - gridOffsetX - gridWidth;

            const finalOffsetX = Math.max(minOffsetX, Math.min(maxOffsetX, offsetX));

            grid.style.transform = `translate(${finalOffsetX}px)`;
        };

        updateTransformation();

        window.addEventListener('resize', updateTransformation);
        const resizeObserver = new ResizeObserver(updateTransformation);

        resizeObserver.observe(container);
        resizeObserver.observe(grid);

        return () => {
            window.removeEventListener('resize', updateTransformation);
            resizeObserver.disconnect();
        };
    };
    return {
        gridCallbackRef,
    };
}
