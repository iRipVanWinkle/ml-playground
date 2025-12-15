import { useMatrixTransform } from './useMatrixTransform';

type MatrixGridProps = {
    size: number;
    topLabel?: string;
    leftLabel?: string;
    children?: React.ReactNode;
};

export function MatrixGridRoot({ size, topLabel, leftLabel, children }: MatrixGridProps) {
    const { matrixGridRef } = useMatrixTransform();

    const style = {
        gridTemplateColumns: `${leftLabel ? 'auto ' : ''}auto repeat(${size}, minmax(50px, 1fr))`,
        gridTemplateRows: `${topLabel ? 'auto ' : ''}auto repeat(${size}, minmax(50px, 1fr))`,
    };

    const bgTransparentStyle =
        topLabel || leftLabel
            ? {
                  gridColumn: `1 / ${leftLabel ? 3 : 2}`,
                  gridRow: `1 / ${topLabel ? 3 : 2}`,
              }
            : undefined;

    return (
        <div className="w-full h-full flex flex-col items-center justify-center overflow-x-auto">
            <div ref={matrixGridRef} className="grid gap-1 p-1" style={style}>
                <div className="bg-transparent" style={bgTransparentStyle}></div>

                {topLabel && (
                    <div
                        className="flex items-center justify-center text-sm text-gray-600 font-normal whitespace-pre"
                        style={{ gridColumn: `${leftLabel ? 3 : 2} / -1` }}
                    >
                        {topLabel}
                    </div>
                )}

                {leftLabel && (
                    <div
                        className="flex items-center justify-center max-w-[20px] text-sm text-gray-600 font-normal whitespace-pre"
                        style={{
                            gridColumn: '1',
                            gridRow: `${topLabel ? 3 : 2} / -1`,
                        }}
                    >
                        <div className="-rotate-90">{leftLabel}</div>
                    </div>
                )}

                {children}
            </div>
        </div>
    );
}

function MatrixColTitle({ children }: { children: string }) {
    return (
        <div
            className="font-medium text-xs min-w-0 max-w-[60px] text-muted-foreground truncate"
            title={children}
        >
            {children}
        </div>
    );
}

function MatrixRowTitle({ children }: { children: string }) {
    return (
        <div
            className="flex items-center font-medium text-xs min-w-0 max-w-[120px] text-muted-foreground"
            title={children}
        >
            <span className="truncate text-right w-full">{children}</span>
        </div>
    );
}

function MatrixCell({ children, className }: { children?: React.ReactNode; className?: string }) {
    return (
        <div
            data-type="matrix-cell"
            className={`
                flex flex-col items-center justify-center
                text-sm font-medium
                aspect-square w-full min-w-[50px] min-h-[50px] max-w-[60px] max-h-[60px]
                border cursor-pointer transition-all duration-200
                hover:scale-105
                rounded-sm
                ${className || ''}
            `}
        >
            {children}
        </div>
    );
}

const MatrixGrid = Object.assign(MatrixGridRoot, {
    ColTitle: MatrixColTitle,
    RowTitle: MatrixRowTitle,
    Cell: MatrixCell,
});

export { MatrixGrid };
