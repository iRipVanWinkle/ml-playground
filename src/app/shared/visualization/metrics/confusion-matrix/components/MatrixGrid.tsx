import { Fragment, useMemo } from 'react';
import { MatrixCell } from './MatrixCell';
import { useMatrixTransform } from '../hooks';

export interface ClassLabel {
    label: string;
    tooltip?: string;
}

interface MatrixGridProps {
    displayMatrix: number[][];
    rowLabels: string[];
    columnLabels: string[];
    classLabels: string[];
}

export function MatrixGrid({
    displayMatrix,
    rowLabels,
    columnLabels,
    classLabels,
}: MatrixGridProps) {
    const size = displayMatrix.length;
    const { gridCallbackRef } = useMatrixTransform({ size });
    const isBinaryClassification = size === 2;

    const rowTotals = useMemo(
        () => displayMatrix.map((row) => row.reduce((sum, value) => sum + value, 0)),
        [displayMatrix],
    );

    return (
        <div className="w-full h-full flex flex-col items-center justify-center overflow-y-auto">
            <div
                ref={gridCallbackRef}
                className="grid gap-1"
                style={{
                    gridTemplateColumns: `auto auto repeat(${size}, minmax(50px, 1fr))`,
                    gridTemplateRows: `auto auto repeat(${size}, minmax(50px, 1fr))`,
                }}
            >
                <div
                    className="bg-transparent"
                    style={{ gridColumn: `1 / 3`, gridRow: `1 / 3` }}
                ></div>

                <div
                    className="flex items-center justify-center text-sm text-gray-600 font-normal whitespace-pre"
                    style={{ gridColumn: `3 / -1` }}
                >
                    Predicted
                </div>

                <div
                    className="flex items-center justify-center max-w-[20px] text-sm text-gray-600 font-normal whitespace-pre"
                    style={{
                        gridColumn: '1',
                        gridRow: `3 / -1`,
                    }}
                >
                    <div className="-rotate-90">Expected</div>
                </div>

                {columnLabels.map((label, index) => (
                    <div
                        key={`col-${index}`}
                        className="flex items-center justify-center font-bold text-xs p-1 min-w-0 max-w-[60px]"
                        title={label}
                    >
                        <span className="truncate w-full">{label}</span>
                    </div>
                ))}

                {displayMatrix.map((row, rowIndex) => (
                    <Fragment key={`row-${rowIndex}`}>
                        <div
                            className="flex items-center font-bold text-xs p-1 min-w-0 max-w-[120px]"
                            title={rowLabels[rowIndex]}
                        >
                            <span className="break-words hyphens-auto leading-tight truncate text-right w-full">
                                {rowLabels[rowIndex]}
                            </span>
                        </div>

                        {row.map((value, colIndex) => (
                            <MatrixCell
                                key={`cell-${rowIndex}-${colIndex}`}
                                value={value}
                                rowTotal={rowTotals[rowIndex]}
                                tooltip={
                                    isBinaryClassification
                                        ? getBinaryTooltip(rowIndex, colIndex, classLabels)
                                        : getTooltip(rowIndex, colIndex, classLabels)
                                }
                                isDiagonal={isDiagonal(rowIndex, colIndex)}
                            />
                        ))}
                    </Fragment>
                ))}
            </div>
        </div>
    );
}

const isDiagonal = (row: number, col: number) => row === col;

const getTooltip = (rowIndex: number, colIndex: number, classLabels: string[]) => {
    const prefix = isDiagonal(rowIndex, colIndex) ? 'Class: ' : 'Error: ';
    return (
        <>
            {prefix} <b>{classLabels[rowIndex]}</b> as <b>{classLabels[colIndex]}</b>
        </>
    );
};

const BINARY_CLASSIFICATION_TOOLTIP = [
    ['True Positives', 'False Positives'],
    ['False Negatives', 'True Negatives'],
];

const getBinaryTooltip = (rowIndex: number, colIndex: number, classLabels: string[]) => {
    const header = BINARY_CLASSIFICATION_TOOLTIP[rowIndex][colIndex];
    return (
        <>
            <h3 className="text-sm font-bold text-center pb-1">{header}</h3>
            {getTooltip(rowIndex, colIndex, classLabels)}
        </>
    );
};
