import * as tf from '@tensorflow/tfjs';
import { afterEach, beforeAll, beforeEach } from 'vitest';

beforeAll(async () => {
    await tf.setBackend('cpu');
    await tf.ready();
});

beforeEach(() => {
    tf.engine().startScope();
});

afterEach(() => {
    tf.engine().endScope();
});
