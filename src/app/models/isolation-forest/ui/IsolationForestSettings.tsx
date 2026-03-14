import type { IsolationForestSettings } from '../types';
import type { ModelSettingsComponentProps } from '@/app/shared/registry';
import { Field, Input, Switch, Label } from '@/app/shared/ui';

const ESTIMATORS_INFO =
    'Number of isolation trees to build. More trees improve stability at the cost of speed.';
const MAX_SAMPLES_INFO = 'Number of samples drawn per tree. Capped at the dataset size.';
const CONTAMINATION_INFO =
    'Expected fraction of anomalies in the data. Used to set the decision threshold.';
const BOOTSTRAP_INFO =
    'When enabled, each tree is built on a bootstrap sample drawn with replacement. When disabled (default), sampling is performed without replacement as in the original paper.';

export function IsolationForestSettings({
    settings,
    disabled,
    onChange,
}: ModelSettingsComponentProps<IsolationForestSettings>) {
    return (
        <>
            <Field label="Estimators" htmlFor="ifEstimators" info={ESTIMATORS_INFO}>
                <Input
                    disabled={disabled}
                    id="ifEstimators"
                    data-testid="if-estimators-input"
                    type="number"
                    min={1}
                    step={1}
                    value={settings.estimators}
                    onChange={(e) => onChange({ estimators: parseInt(e.target.value) || 1 })}
                />
            </Field>

            <Field label="Max Samples" htmlFor="ifMaxSamples" info={MAX_SAMPLES_INFO}>
                <Input
                    disabled={disabled}
                    id="ifMaxSamples"
                    data-testid="if-max-samples-input"
                    type="number"
                    min={1}
                    step={1}
                    value={settings.maxSamples}
                    onChange={(e) => onChange({ maxSamples: parseInt(e.target.value) || 1 })}
                />
            </Field>

            <Field label="Contamination" htmlFor="ifContamination" info={CONTAMINATION_INFO}>
                <Input
                    disabled={disabled}
                    id="ifContamination"
                    data-testid="if-contamination-input"
                    type="number"
                    min={0}
                    max={0.5}
                    step={0.01}
                    value={settings.contamination}
                    onChange={(e) => onChange({ contamination: parseFloat(e.target.value) || 0 })}
                />
            </Field>

            <Field label="Bootstrap" htmlFor="ifBootstrap" info={BOOTSTRAP_INFO}>
                <div className="flex items-center gap-2">
                    <Switch
                        id="ifBootstrap"
                        data-testid="if-bootstrap-switch"
                        disabled={disabled}
                        checked={settings.bootstrap}
                        onCheckedChange={(checked) => onChange({ bootstrap: checked })}
                    />
                    <Label htmlFor="ifBootstrap" className="font-normal">
                        {settings.bootstrap ? 'Enabled' : 'Disabled'}
                    </Label>
                </div>
            </Field>
        </>
    );
}
