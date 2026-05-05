import { Fragment } from 'react';
import { MatrixCell } from './MatrixCell';
import { MatrixGrid as BaseMatrixGrid } from '../../../base';

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
    const isBinaryClassification = size === 2;

    const rowTotals = displayMatrix.map((row) => row.reduce((sum, value) => sum + value, 0));

    return (
        <BaseMatrixGrid size={size} topLabel="Predicted" leftLabel="Expected">
            {columnLabels.map((label, index) => (
                <BaseMatrixGrid.ColTitle key={`col-${index}`}>{label}</BaseMatrixGrid.ColTitle>
            ))}

            {displayMatrix.map((row, rowIndex) => (
                <Fragment key={`row-${rowIndex}`}>
                    <BaseMatrixGrid.RowTitle>{rowLabels[rowIndex]}</BaseMatrixGrid.RowTitle>
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
        </BaseMatrixGrid>
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
