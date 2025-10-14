import type { CallbackParameters, ModelRepresentation, ModelSettings } from '@/app/models/types';

type ModelSettingsMap = {
    [M in ModelSettings as M['type']]: M;
};

export type SettingsOf<K extends keyof ModelSettingsMap> = ModelSettingsMap[K];

type ModelRepresentationMap = {
    [M in ModelRepresentation as M['type']]: M['representation'];
};

export type RepresentationOf<K extends keyof ModelRepresentationMap> = ModelRepresentationMap[K];

type CallbackParametersMap = {
    [M in CallbackParameters as M['type']]: M['callbackParameters'];
};

export type CallbackParametersOf<K extends keyof CallbackParametersMap> = CallbackParametersMap[K];
