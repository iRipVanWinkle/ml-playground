import type { NaiveBayesParams } from '@/ml/types';
import { ImageDisplay } from './image-display';

type ImageParametersProps = {
    params: NaiveBayesParams;
    categories?: string[];
    selectedClassIndex?: number;
};

export function ImageParameters({ params, categories, selectedClassIndex }: ImageParametersProps) {
    const classes = selectedClassIndex !== undefined ? [selectedClassIndex] : params.classes;

    return (
        <div className="flex flex-col gap-3">
            {classes.map((classIndex) => {
                const categoryName = categories?.[classIndex] || `Class ${classIndex}`;
                return (
                    <ImageDisplay
                        key={classIndex}
                        params={params}
                        classIndex={classIndex}
                        categoryName={categoryName}
                    />
                );
            })}
        </div>
    );
}
