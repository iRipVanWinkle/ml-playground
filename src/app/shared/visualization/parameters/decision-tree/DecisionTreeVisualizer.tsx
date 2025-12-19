import { useState } from 'react';
import { TreeSVG } from './components/TreeSVG';
import { RawTreeText } from './components/RawTreeText';
import { DecisionTreeContext } from './DecisionTreeContext';
import type { TreeNode } from '@/ml/types';
import { ViewSelector } from './components/ViewSelector';
import { TreeSelector } from './components/TreeSelector';

interface DecisionTreeVisualizerProps {
    trees: ReadonlyArray<TreeNode>;
    featureLabels: string[];
    categories?: string[];
}

export function DecisionTreeVisualizer({
    trees,
    featureLabels,
    categories,
}: DecisionTreeVisualizerProps) {
    const [selectedTreeIndex, setSelectedTreeIndex] = useState(0);
    const [viewMode, setViewMode] = useState('graph');

    const selectedTree = trees[selectedTreeIndex];

    return (
        <DecisionTreeContext.Provider value={{ featureLabels, categories }}>
            <div className="flex items-center justify-between px-4 mb-4">
                <h3 className="text-lg font-semibold">Learned Tree Structure</h3>

                <div className="flex items-center justify-end gap-4">
                    {trees.length > 1 && (
                        <TreeSelector
                            amount={trees.length}
                            value={selectedTreeIndex}
                            onChange={setSelectedTreeIndex}
                        />
                    )}
                    <ViewSelector value={viewMode} onChange={setViewMode} />
                </div>
            </div>

            <div className="flex flex-col h-full w-full gap-4">
                <div className="flex-1 w-full min-h-0 rounded-lg overflow-hidden relative">
                    {viewMode === 'graph' && <TreeSVG tree={selectedTree} />}
                    {viewMode === 'raw' && (
                        <RawTreeText
                            tree={selectedTree}
                            featureLabels={featureLabels}
                            categories={categories}
                        />
                    )}
                </div>
            </div>
        </DecisionTreeContext.Provider>
    );
}
