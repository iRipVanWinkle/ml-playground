import { Maximize, MinusIcon, PlusIcon } from 'lucide-react';
import type { TreeNode as MLTreeNode } from '@/ml/types';
import { Button, ButtonGroup } from '@/app/shared/ui';
import { usePlotlyColors } from '../../../colors';
import { TreeNode } from './TreeNode';
import { TreeEdge } from './TreeEdge';
import {
    useTreeLayout,
    useContainerDimensions,
    useTransformControls,
    usePanZoomInteractions,
    useEdgePaths,
} from '../hooks';
import type { LayoutNode } from '../types';
import { memo } from 'react';

interface TreeSVGProps {
    tree: MLTreeNode;
}

export function TreeSVG({ tree }: TreeSVGProps) {
    const { paperBg, gridColor } = usePlotlyColors();

    const { containerRef, width, height } = useContainerDimensions();

    const { nodes, edges, bounds } = useTreeLayout({ tree, width });

    const { transformString, setTransform, zoomIn, zoomOut, reset } = useTransformControls({
        nodes,
        width,
        height,
        bounds,
    });

    usePanZoomInteractions({
        containerRef,
        setTransform,
    });

    const edgePaths = useEdgePaths(edges);

    return (
        <div
            ref={containerRef}
            className="w-full h-full overflow-hidden border rounded-lg relative cursor-move"
            style={{ height: height, backgroundColor: paperBg, borderColor: gridColor }}
        >
            <div className="absolute bottom-2 right-2 flex flex-col gap-1">
                <ButtonGroup orientation="horizontal" aria-label="Media controls" className="h-fit">
                    <Button onClick={zoomIn} variant="outline" size="icon">
                        <PlusIcon />
                    </Button>
                    <Button onClick={reset} variant="outline" size="icon" aria-label="Reset view">
                        <Maximize />
                    </Button>
                    <Button onClick={zoomOut} variant="outline" size="icon">
                        <MinusIcon />
                    </Button>
                </ButtonGroup>
            </div>

            <svg width="100%" height="100%" className="select-none">
                <g transform={transformString}>
                    <Elements nodes={nodes} edgePaths={edgePaths} />
                </g>
            </svg>
        </div>
    );
}

type ElementsProps = {
    nodes: LayoutNode[];
    edgePaths: {
        id: string;
        path: string;
        labelX: number;
        labelY: number;
        label?: string | undefined;
    }[];
};

const Elements = memo(function Elements({ nodes, edgePaths }: ElementsProps) {
    return (
        <>
            {/* Edges */}
            {edgePaths.map((edge) => (
                <TreeEdge key={edge.id} edge={edge} />
            ))}

            {/* Nodes */}
            {nodes.map((node) => (
                <TreeNode key={node.id} node={node} />
            ))}
        </>
    );
});
