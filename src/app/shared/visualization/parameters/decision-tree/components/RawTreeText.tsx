import type { TreeNode } from '@/ml/types';
import { generateRawModelText } from '../utils';

interface RawTreeTextProps {
    tree: TreeNode;
    featureLabels: string[];
    categories?: string[];
}

export function RawTreeText({ tree, featureLabels, categories }: RawTreeTextProps) {
    const rawText = generateRawModelText(tree, featureLabels, categories);

    return (
        <div className="rounded-lg border bg-muted/50 p-4 text-xs text-left">
            <pre className="whitespace-pre">{rawText}</pre>
        </div>
    );
}
