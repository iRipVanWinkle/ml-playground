import { useLayoutEffect, useRef, useState } from 'react';

type Size = { width: number; height: number };

export function useContainerSize(initial: Size) {
    const containerRef = useRef<HTMLDivElement>(null);
    const [size, setSize] = useState<Size>(initial);

    useLayoutEffect(() => {
        const el = containerRef.current;
        if (!el) return;
        const ro = new ResizeObserver((entries) => {
            const entry = entries[0];
            if (!entry) return;
            const { width, height } = entry.contentRect;
            if (width > 0 && height > 0) setSize({ width, height });
        });
        ro.observe(el);
        return () => ro.disconnect();
    }, []);

    return { containerRef, size };
}
