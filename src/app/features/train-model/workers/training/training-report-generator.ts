import { encode } from '../../helpers/float32Array';

export class TrainingReportGenerator {
    generateReport(
        liveResults: Record<string, number | number[] | number[][] | string | undefined>,
    ): Float32Array {
        const report = Object.entries(liveResults).reduce(
            (acc, [key, value]) => ({
                ...acc,
                ...(typeof value !== 'string' ? { [key]: value } : {}),
            }),
            {},
        );

        return encode(report);
    }
}
