import { useContext } from 'react';
import { useColor, usePlotlyColors } from '../../../colors';
import { DecisionTreeContext } from '../DecisionTreeContext';
import type { LayoutNode } from '../types';
import { NODE_HEIGHT, NODE_WIDTH } from '../constants';
interface TreeNodeProps {
    node: LayoutNode;
    width?: number;
    height?: number;
}

export function TreeNode({ node, width = NODE_WIDTH, height = NODE_HEIGHT }: TreeNodeProps) {
    const { x, y, data, isLeaf } = node;
    const { featureIndex = 0, threshold = 0, value = 0 } = data;

    if (isLeaf) {
        return <Leaf value={value} x={x} y={y} width={width} height={height} />;
    }

    return (
        <Node
            featureIndex={featureIndex}
            threshold={threshold}
            x={x}
            y={y}
            width={width}
            height={height}
        />
    );
}

type LeafProps = {
    width: number;
    height: number;
    x: number;
    y: number;
    value: number;
};

function Leaf({ width = NODE_WIDTH, height = NODE_HEIGHT, x, y, value }: LeafProps) {
    const { categories } = useContext(DecisionTreeContext);
    const { getColor } = useColor();
    const { textColor } = usePlotlyColors();

    const color = categories ? getColor(value) : getColor(0);

    const displayValue = categories?.[value] ?? value.toFixed(3);

    return (
        <g transform={`translate(${x - width / 2}, ${y - height / 2})`}>
            <rect
                width={width}
                height={height}
                rx={8}
                ry={8}
                fill="none"
                stroke={color}
                strokeWidth={2}
            />
            <text
                x={width / 2}
                y={height / 2}
                dy=".3em"
                textAnchor="middle"
                fill={textColor}
                fontSize="14"
                fontWeight="bold"
                style={{ pointerEvents: 'none' }}
            >
                {displayValue}
            </text>
        </g>
    );
}

type NodeProps = {
    width: number;
    height: number;
    x: number;
    y: number;
    featureIndex: number;
    threshold: number;
};

function Node({
    width = NODE_WIDTH,
    height = NODE_HEIGHT,
    x,
    y,
    featureIndex,
    threshold,
}: NodeProps) {
    const { featureLabels } = useContext(DecisionTreeContext);
    const { textColor, axisLineColor } = usePlotlyColors();

    const featureName =
        featureIndex != null
            ? (featureLabels?.[featureIndex] ?? `Feature ${featureIndex}`)
            : undefined;
    return (
        <g transform={`translate(${x}, ${y})`}>
            <rect
                x={-width / 2}
                y={-height / 2}
                width={width}
                height={height}
                rx={8}
                ry={8}
                fill="none"
                stroke={axisLineColor}
                strokeWidth={2}
            />
            <text textAnchor="middle" fill={textColor} style={{ pointerEvents: 'none' }}>
                <tspan x="0" dy="-0.5em">
                    {featureName}
                </tspan>
                <tspan x="0" dy="2em" className="font-semibold">
                    ≤ {threshold?.toFixed(3) ?? '?'}
                </tspan>
            </text>
        </g>
    );
}
