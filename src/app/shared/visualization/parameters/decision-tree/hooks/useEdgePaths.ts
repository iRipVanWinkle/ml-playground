import { useMemo } from 'react';
import { EDGE_OFFSET } from '../constants';
import type { LayoutEdge } from '../types';

export function useEdgePaths(edges: LayoutEdge[]) {
    return useMemo(
        () =>
            edges.map((edge) => {
                const sx = edge.source.x;
                const sy = edge.source.y + EDGE_OFFSET;
                const tx = edge.target.x;
                const ty = edge.target.y - EDGE_OFFSET;

                return {
                    id: edge.id,
                    path: `M ${sx} ${sy} L ${tx} ${ty}`,
                    labelX: (sx + tx) / 2,
                    labelY: (sy + ty) / 2,
                    label: edge.label,
                };
            }),
        [edges],
    );
}
