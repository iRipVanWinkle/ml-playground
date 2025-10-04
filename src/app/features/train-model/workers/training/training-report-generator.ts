import { encode } from '../../helpers/float32Array';
import type { LiveResults } from './live-metrics';
import type { TrainingSession } from './training-session';

export interface TrainerReport {
    trainLossHistory: number[][];
    trainAccuracy: number;
    testAccuracy: number;
    testLoss: number;
    iterations: number[];
    trainPredictedLabels: number[][];
    testPredictedLabels: number[][];
    predictionPredictedLabels: number[][];
    theta: number[][];
}

export class TrainingReportGenerator {
    generateReport(liveResults: LiveResults, session: TrainingSession): Float32Array {
        const report = {
            trainLossHistory: session.getFormattedLossHistory(),
            iterations: session.getIterations(),
            trainAccuracy: liveResults.trainAccuracy!,
            testAccuracy: liveResults.testAccuracy!,
            testLoss: liveResults.testLoss!,
            trainPredictedLabels: liveResults.trainPredictedLabels ?? [],
            testPredictedLabels: liveResults.testPredictedLabels ?? [],
            predictionPredictedLabels: liveResults.predictionPredictedLabels ?? [],
            theta: liveResults.thetaArray ?? [],
        };

        return encode(report);
    }
}
