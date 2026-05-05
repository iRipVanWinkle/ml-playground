import { useRef, useEffect, type RefObject } from 'react';
import { MAX_SCALE, MIN_SCALE } from '../constants';

interface Transform {
    x: number;
    y: number;
    scale: number;
}

interface UsePanZoomInteractionsProps {
    containerRef: RefObject<HTMLDivElement | null>;
    setTransform: React.Dispatch<React.SetStateAction<Transform>>;
}

export function usePanZoomInteractions({
    containerRef,
    setTransform,
}: UsePanZoomInteractionsProps) {
    const isDragging = useRef(false);
    const lastMousePos = useRef({ x: 0, y: 0 });

    useEffect(() => {
        const container = containerRef.current;
        if (!container) return;

        const handleWheel = (e: WheelEvent) => {
            e.preventDefault();

            const scaleAmount = -e.deltaY * 0.001;
            setTransform((prev) => ({
                ...prev,
                scale: Math.min(Math.max(MIN_SCALE, prev.scale + scaleAmount), MAX_SCALE),
            }));
        };

        const handleMouseDown = (e: MouseEvent) => {
            isDragging.current = true;
            lastMousePos.current = { x: e.clientX, y: e.clientY };
            e.preventDefault();
        };

        const handleMouseMove = (e: MouseEvent) => {
            if (!isDragging.current) return;

            const dx = e.clientX - lastMousePos.current.x;
            const dy = e.clientY - lastMousePos.current.y;

            setTransform((prev) => ({
                ...prev,
                x: prev.x + dx,
                y: prev.y + dy,
            }));

            lastMousePos.current = { x: e.clientX, y: e.clientY };
        };

        const handleMouseUp = () => {
            isDragging.current = false;
        };

        container.addEventListener('mousedown', handleMouseDown);
        document.addEventListener('mousemove', handleMouseMove);
        document.addEventListener('mouseup', handleMouseUp);
        document.addEventListener('mouseleave', handleMouseUp);
        // Wheel event listener with passive: false to allow preventDefault
        container.addEventListener('wheel', handleWheel, { passive: false });
        return () => {
            container.removeEventListener('mousedown', handleMouseDown);
            document.removeEventListener('mousemove', handleMouseMove);
            document.removeEventListener('mouseup', handleMouseUp);
            document.removeEventListener('mouseleave', handleMouseUp);
            container.removeEventListener('wheel', handleWheel);
        };
    }, [containerRef, setTransform]);
}
