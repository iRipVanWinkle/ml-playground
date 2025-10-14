import type { ModelType } from '@/app/models/types';
import type { TaskType } from '../types';
import type { ModelDefinition } from './types/model-definition';

export type ModelDefinitionsMap = {
    [K in ModelType]: ModelDefinition<K>;
};

export class ModelRegistry {
    private registrations: Partial<ModelDefinitionsMap>;

    constructor(registrations: Array<ModelDefinitionsMap[ModelType]>) {
        this.registrations = Object.fromEntries(registrations.map((r) => [r.key, r]));
    }

    getForTask(taskType: TaskType): ModelDefinitionsMap[keyof ModelDefinitionsMap][] {
        return Object.values(this.registrations).filter((r) => r.taskTypes.includes(taskType));
    }

    get<T extends ModelType>(modelId: T): ModelDefinition<T> {
        const registration = this.registrations[modelId];

        if (!registration) {
            throw new Error(`Model with id "${modelId}" is not registered.`);
        }

        return registration;
    }
}
