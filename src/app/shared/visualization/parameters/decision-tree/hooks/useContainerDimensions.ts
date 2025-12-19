import { useRef, useState, useLayoutEffect } from 'react';

export function useContainerDimensions() {
    const containerRef = useRef<HTMLDivElement>(null);
    const [dimensions, setDimensions] = useState({ width: 800, height: 600 });

    useLayoutEffect(() => {
        if (containerRef.current) {
            const { offsetWidth, offsetHeight } = containerRef.current;
            setDimensions({ width: offsetWidth, height: offsetHeight });
        }
    }, []);

    return {
        containerRef,
        width: dimensions.width,
        height: dimensions.height,
    };
}
