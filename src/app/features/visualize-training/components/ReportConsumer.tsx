import type { TrainingReport } from '@/app/models/types';
import { useTrainingReport } from '@/app/store';

type ReportConsumerProps = {
    children: (report: TrainingReport) => React.ReactNode;
};

export function ReportConsumer({ children }: ReportConsumerProps) {
    const report = useTrainingReport();

    return children(report);
}
