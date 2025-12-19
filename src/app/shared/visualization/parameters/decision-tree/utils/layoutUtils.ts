import type { LayoutNode, TreeBounds } from '../types';
import { NODE_HALF_W, NODE_HALF_H, DEFAULT_SCALE } from '../constants';

export const calculateAutoFit = (
    nodes: LayoutNode[],
    viewportWidth: number,
    viewportHeight: number,
    bounds?: TreeBounds,
) => {
    // Dynamic calculation based on viewport
    const padding = Math.min(viewportWidth, viewportHeight) * 0.025;
    const verticalOffset = viewportHeight * 0.05;
    const defaultScale = DEFAULT_SCALE;

    if (nodes.length === 0) return { x: 0, y: 0, scale: 1 };

    let treeWidth: number;
    let treeHeight: number;
    let treeCenterX: number;
    let treeCenterY: number;

    if (bounds) {
        treeWidth = bounds.width;
        treeHeight = bounds.height;
        treeCenterX = (bounds.minX + bounds.maxX) / 2;
        treeCenterY = (bounds.minY + bounds.maxY) / 2;
    } else {
        // Calculate bounds of the tree nodes dynamically
        let minX = Infinity;
        let maxX = -Infinity;
        let minY = Infinity;
        let maxY = -Infinity;

        nodes.forEach((node) => {
            minX = Math.min(minX, node.x - NODE_HALF_W);
            maxX = Math.max(maxX, node.x + NODE_HALF_W);
            minY = Math.min(minY, node.y - NODE_HALF_H);
            maxY = Math.max(maxY, node.y + NODE_HALF_H);
        });

        treeWidth = maxX - minX;
        treeHeight = maxY - minY;
        treeCenterX = (minX + maxX) / 2;
        treeCenterY = (minY + maxY) / 2;
    }

    if (treeWidth === 0 || treeHeight === 0) {
        return { x: 0, y: 0, scale: 1 };
    }

    // Calculate scale to fit
    const availableWidth = viewportWidth - padding * 2;
    const availableHeight = viewportHeight - padding * 2;

    const scaleX = availableWidth / treeWidth;
    const scaleY = availableHeight / treeHeight;

    const newScale = Math.min(scaleX, scaleY, defaultScale);

    return {
        x: -treeCenterX * newScale,
        y: viewportHeight / 2 - verticalOffset - treeCenterY * newScale,
        scale: newScale,
    };
};
