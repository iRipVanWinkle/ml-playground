import { useState, useEffect, useRef, startTransition } from 'react';
import { calculateAutoFit } from '../utils';
import { MAX_SCALE, MIN_SCALE, VERTICAL_OFFSET } from '../constants';
import type { LayoutNode, TreeBounds } from '../types';

interface UseTransformControlsProps {
    nodes: LayoutNode[];
    width: number;
    height: number;
    bounds: TreeBounds;
}

export function useTransformControls({ nodes, width, height, bounds }: UseTransformControlsProps) {
    const prevNodesKeyRef = useRef<string>('');
    const prevWidthRef = useRef(width);
    const prevHeightRef = useRef(height);

    const calculateAutoFitTransform = () => calculateAutoFit(nodes, width, height, bounds);

    const [transform, setTransform] = useState(() => calculateAutoFitTransform());

    const nodesKey = nodes.map((n) => n.id).join(',');
    useEffect(() => {
        const nodesChanged = prevNodesKeyRef.current !== nodesKey;
        const dimensionsChanged =
            prevWidthRef.current !== width || prevHeightRef.current !== height;

        if (nodesChanged || dimensionsChanged) {
            prevNodesKeyRef.current = nodesKey;
            prevWidthRef.current = width;
            prevHeightRef.current = height;

            startTransition(() => setTransform(calculateAutoFit(nodes, width, height, bounds)));
        }
    }, [nodesKey, width, height, nodes, bounds]);

    const { x, y, scale } = transform;
    const transformString = `translate(${x + width / 2}, ${y + VERTICAL_OFFSET}) scale(${scale})`;

    const handleZoomIn = () => {
        setTransform(({ scale, ...rest }) => ({
            ...rest,
            scale: Math.min(MAX_SCALE, scale + 0.1),
        }));
    };

    const handleZoomOut = () => {
        setTransform(({ scale, ...rest }) => ({
            ...rest,
            scale: Math.max(MIN_SCALE, scale - 0.1),
        }));
    };

    const handleReset = () => {
        setTransform(calculateAutoFitTransform());
    };

    return {
        transform,
        transformString,
        setTransform,
        zoomIn: handleZoomIn,
        zoomOut: handleZoomOut,
        reset: handleReset,
    };
}
