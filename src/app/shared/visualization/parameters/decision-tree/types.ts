export type LayoutNode = {
    x: number;
    y: number;
    data: TreeNodeData;
    id: string; // Unique ID for key/tracking
    isLeaf: boolean;
};

export type LayoutEdge = {
    id: string;
    source: LayoutNode;
    target: LayoutNode;
    label?: string; // e.g. "yes" / "no"
};

export interface TreeNodeData {
    id: string;
    value?: number;
    threshold?: number;
    featureIndex?: number;
    isLeaf: boolean;
    children?: TreeNodeData[];
}

export interface TreeBounds {
    minX: number;
    maxX: number;
    minY: number;
    maxY: number;
    width: number;
    height: number;
}
