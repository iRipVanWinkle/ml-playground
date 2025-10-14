import type { ModelType } from '@/app/models/types';
import type { WorkerDefinition } from './types/worker-definition';

export type WorkerDefinitionsMap = {
    [K in ModelType]: WorkerDefinition<K>;
};

export class WorkerRegistry {
    private registrations: Partial<WorkerDefinitionsMap>;

    constructor(registrations: Array<WorkerDefinitionsMap[keyof WorkerDefinitionsMap]>) {
        this.registrations = Object.fromEntries(registrations.map((r) => [r.key, r]));
    }

    get<T extends ModelType>(workerId: T): WorkerDefinition<T> {
        const registration = this.registrations[workerId];

        if (!registration) {
            throw new Error(`Worker with id "${workerId}" is not registered.`);
        }

        return registration;
    }
}
