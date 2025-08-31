import { concat, Tensor, type Tensor2D } from '@tensorflow/tfjs';
import type {
    CallbackParameters,
    ModelRepresentation,
    OptimizerCallbackParameters,
    TreeCallbackParameters,
    TreeNode,
} from '@/ml/types';

function fixLength(matrix: number[][]): number[][] {
    const minLength = Math.min(...matrix.map((m) => m.length));
    return matrix.map((m) => m.slice(0, minLength));
}

function isTensorParameter(param: ModelRepresentation | null): param is Tensor2D {
    return param instanceof Tensor && param.rank === 2;
}

function isOptimizerCallbackParameters(
    param: CallbackParameters,
): param is OptimizerCallbackParameters {
    return 'threadId' in param && 'iteration' in param && 'theta' in param && 'loss' in param;
}

function isTreeCallbackParameters(param: CallbackParameters): param is TreeCallbackParameters {
    return 'threadId' in param && 'iteration' in param && 'tree' in param;
}

export class TrainingSession {
    private modelRepresentation: ModelRepresentation | null = null;
    private lossHistory: number[][] = [];
    private iterationCounts: number[] = [];
    private thetaArray: Tensor2D[] | TreeNode[] = [];

    constructor({ numThreads }: { numThreads: number }) {
        this.lossHistory = Array.from({ length: numThreads }, () => []);
        this.iterationCounts = Array.from({ length: numThreads }, () => 0);
    }

    updateIteration(params: CallbackParameters): void {
        const { threadId, iteration } = params;

        if (isOptimizerCallbackParameters(params)) {
            const { theta, loss } = params;

            this.thetaArray[threadId] = theta;
            if (isTensorParameter(this.modelRepresentation)) {
                this.modelRepresentation.dispose();
            }
            this.modelRepresentation = concat(this.thetaArray.filter(Boolean), 1) as Tensor2D;
            this.lossHistory[threadId].push(loss);
        } else if (isTreeCallbackParameters(params)) {
            const { tree } = params;

            this.thetaArray[threadId] = tree;
            this.modelRepresentation = this.thetaArray.filter(Boolean) as TreeNode[];
        }

        this.iterationCounts[threadId] = iteration + 1;
    }

    getIterations(): number[] {
        return [...this.iterationCounts];
    }

    getModelRepresentation(): ModelRepresentation {
        return this.modelRepresentation!;
    }

    getFormattedLossHistory(): number[][] {
        return fixLength(this.lossHistory);
    }

    dispose(): void {
        this.thetaArray.forEach((theta) => theta instanceof Tensor && theta.dispose());
    }
}
