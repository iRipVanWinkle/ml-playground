import { usePlotlyColors } from '../../../colors';

interface TreeEdgeProps {
    edge: {
        id: string;
        path: string;
        labelX: number;
        labelY: number;
        label?: string;
    };
}

export function TreeEdge({ edge }: TreeEdgeProps) {
    const { gridColor, textColor } = usePlotlyColors();

    return (
        <g>
            <path d={edge.path} stroke={gridColor} strokeWidth="2" />
            {edge.label && (
                <text
                    x={edge.labelX}
                    y={edge.labelY}
                    dy="-5"
                    textAnchor="middle"
                    fill={textColor}
                    fontSize="12"
                    fontStyle="italic"
                >
                    {edge.label}
                </text>
            )}
        </g>
    );
}
