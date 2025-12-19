import { useMemo } from 'react';
import type { TreeNode } from '@/ml/types';
import type { LayoutNode, LayoutEdge, TreeNodeData } from '../types';
import { LEVEL_HEIGHT, NODE_HEIGHT, NODE_SPACING, NODE_WIDTH, ROOT_Y_OFFSET } from '../constants';

interface TreeLayoutProps {
    tree: TreeNode;
    width: number;
}

export function useTreeLayout({ tree, width }: TreeLayoutProps) {
    return useMemo(() => {
        const nodes: LayoutNode[] = [];
        const edges: LayoutEdge[] = [];

        // Helper to convert TreeNode to TreeNodeData with ID
        const transformNode = (node: TreeNode, id = '0'): TreeNodeData => {
            const isLeaf = !node.leftChild && !node.rightChild;
            const children: TreeNodeData[] = [];

            if (node.leftChild) {
                children.push(transformNode(node.leftChild, `${id}-L`));
            }
            if (node.rightChild) {
                children.push(transformNode(node.rightChild, `${id}-R`));
            }

            return {
                id,
                value: node.value,
                threshold: node.threshold ?? undefined,
                featureIndex: node.featureIndex ?? undefined,
                isLeaf,
                children,
            };
        };

        const rootData = transformNode(tree);

        // Simple layout algorithm
        // 1. Calculate required width for each subtree
        // 2. Position nodes

        // Map of node ID to its subtree width - computed in single pass
        const subtreeWidths = new Map<string, number>();

        const computeWidths = (node: TreeNodeData): number => {
            if (!node.children || node.children.length === 0) {
                const width = NODE_WIDTH + NODE_SPACING;
                subtreeWidths.set(node.id, width);
                return width;
            }
            const width = node.children.reduce((sum, child) => sum + computeWidths(child), 0);
            subtreeWidths.set(node.id, width);
            return width;
        };

        computeWidths(rootData);

        // Map to store node references for edge creation
        const nodeMap = new Map<string, LayoutNode>();

        // Initialize bounds
        let minX = Infinity;
        let maxX = -Infinity;
        let minY = Infinity;
        let maxY = -Infinity;

        const effectiveNodeWidth = NODE_WIDTH;
        const effectiveNodeHeight = NODE_HEIGHT;

        const assignCoordinates = (node: TreeNodeData, x: number, y: number, level: number) => {
            const layoutNode: LayoutNode = {
                x,
                y,
                data: node,
                id: node.id,
                isLeaf: node.isLeaf,
            };
            nodes.push(layoutNode);
            nodeMap.set(node.id, layoutNode);

            // Update bounds
            minX = Math.min(minX, x - effectiveNodeWidth / 2);
            maxX = Math.max(maxX, x + effectiveNodeWidth / 2);
            minY = Math.min(minY, y - effectiveNodeHeight / 2);
            maxY = Math.max(maxY, y + effectiveNodeHeight / 2);

            if (node.children && node.children.length > 0) {
                const totalWidth = subtreeWidths.get(node.id) || 0;
                let currentX = x - totalWidth / 2;

                node.children.forEach((child, index) => {
                    const childWidth = subtreeWidths.get(child.id) || 0;
                    const childX = currentX + childWidth / 2;
                    const childY = y + LEVEL_HEIGHT;

                    // Create edge after child node is created (in next recursive call)
                    // We'll create it when processing the child
                    assignCoordinates(child, childX, childY, level + 1);

                    // Now create edge using the stored node references
                    const childNode = nodeMap.get(child.id);
                    if (childNode) {
                        edges.push({
                            id: `${node.id}-${child.id}`,
                            source: layoutNode,
                            target: childNode,
                            label: index === 0 ? 'yes' : 'no', // Assuming left is yes, right is no
                        });
                    }

                    currentX += childWidth;
                });
            }
        };

        // Center the root
        assignCoordinates(rootData, width / 2, ROOT_Y_OFFSET, 0);

        const bounds = {
            minX: minX === Infinity ? 0 : minX,
            maxX: maxX === -Infinity ? 0 : maxX,
            minY: minY === Infinity ? 0 : minY,
            maxY: maxY === -Infinity ? 0 : maxY,
            width: maxX - minX,
            height: maxY - minY,
        };

        return { nodes, edges, bounds };
    }, [tree, width]);
}
